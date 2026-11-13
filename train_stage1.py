"""
ReFuSe Stage 1: region-level fusion of the source model and the VLM.

One script for every setting; the experiment config (configs/*.yaml) picks
  source    gta5 | synthia     (synthia: terrain/truck/train are switched off)
  backbone  deeplabv2 | mitb5  (DeepLabV2-ResNet101 or MiT-B5 + DAFormer head)
  vlm       corrclip | proxyclip

The target model starts from the source-only model and is trained on the
unlabelled target training images (Cityscapes train). No target label is
read anywhere.

For every image and every SAM2 region R (valid pixels only, >= 50 pixels):
  y_s = majority of the reference label (epoch 1: frozen source pseudo-label
        Ps; later epochs: the model's own argmax), y_c = majority of the VLM
        label, a = fraction of pixels where the two agree.
  q_t = (exp(-H(soften(Pt_mean))) + cons_t + bdry_t) / 3,
  q_c = (exp(-H(Pc_mean))       + cons_c + bdry_c) / 3
        (confidence, pixel consensus, boundary agreement).
  Thresholds theta_agree / theta_rel / theta_margin = percentiles (60/30/25)
  of a, max(q_t,q_c), |q_t-q_c| over 50 calibration images (thresholds.json).
  Case 1  y_s == y_c and a > theta_agree           -> hard label y_s
  Case 2  y_s != y_c, max(q) > theta_rel, |q_t-q_c| > theta_margin
                                                    -> soft label: the more
          reliable side's mean distribution, sharpened (tau 0.5)
  Case 3  otherwise                                 -> no supervision
Additions (each with its own switch):
  (A) class-balanced Case 1 CE                      --cb-power (0 = off)
  (B) text z-score verification of Case 1 labels   --text-verify on|off
      (rejected region: hard label = confuser if the gap is large, otherwise
       partial label L_partial = -log sum_{y, confusers} Pt)
  (C) VLM-blind classes (classes the VLM almost never predicts but the source
      does): hard label from Ps if q_src > --blind-q-thresh  --blind-classes
L = L_CE_agree + lambda_d L_CE_disagree + lambda_p L_partial + mu L_balance.
STEP numbers in the code: 1 skip small regions, 2 cap the pixels used for the
region statistics (5000; the loss still covers the whole region), 3 majority
votes, 4 agreement, 5 reliability q, 6 threshold calibration, 7 case decision,
8 losses.

Model selection (no target labels): 200 training images are held out; the
checkpoint with the lowest disagreement rate between the model's argmax and
the VLM label on them is best.pth; training stops after 5 epochs without a
new minimum.

Usage:
    python train_stage1.py --config configs/gta5_deeplabv2_corrclip.yaml --gpu 0
    python train_stage1.py --config ... --start-epoch 6      # resume from dsp_epoch5.pth
    python train_stage1.py --config ... --debug --run-name debug_stage1
Outputs: <work_dir>/<run_name>/ (default run name: stage1).
"""

import os
import sys
import json
import math
import argparse
from collections import defaultdict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ── ReFuSe modules ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refuse.common import (NUM_CLASSES, IGNORE_LABEL, TARGET_H, TARGET_W,  # noqa: E402
                           CITYSCAPES_CLASSES, colorize)
from refuse.backbones import freeze_bn  # noqa: E402
from refuse.config import load_config, make_backbone, default_text_z_classes  # noqa: E402
from refuse.vlm import VLM  # noqa: E402

PS_SUFFIX   = '_pseudo_label.png'
IMG_SUFFIX  = '_leftImg8bit.png'
MASK_SUFFIX = '_leftImg8bit.npz'
EPS = 1e-8

# Set once in main() from the experiment config.
BB = None                  # refuse.backbones.Backbone
VLM_T = None               # refuse.vlm.VLM (the VLM teacher)
ABSENT_CLASSES = []        # classes the source domain does not have (SYNTHIA: terrain, truck, train)
ABSENT_T = None
VAL_IMG_ROOT = None        # Cityscapes val images, for the image-only sanity panels


def preprocess(img_bgr):
    return BB.preprocess(img_bgr)


def predict(model, img_bgr, device):
    return BB.predict(model, img_bgr, device)


def collect_entries(ps_root, img_root, mask_root):
    """Every training image that has a source pseudo-label PNG (epoch-1
    reference, VLM-blind anchor), a SAM2 instance mask and the VLM files it
    needs (pre-extracted VLM outputs; nothing for ProxyCLIP)."""
    entries = []
    for city in sorted(os.listdir(img_root)):
        img_city = os.path.join(img_root, city)
        if not os.path.isdir(img_city):
            continue
        for fname in sorted(os.listdir(img_city)):
            if not fname.endswith(IMG_SUFFIX):
                continue
            stem = fname[:-len(IMG_SUFFIX)]
            e = {'stem': stem, 'city': city,
                 'ps_path': os.path.join(ps_root, city, stem + PS_SUFFIX),
                 'mask_path': os.path.join(mask_root, city, stem + MASK_SUFFIX),
                 'img_path': os.path.join(img_city, fname),
                 **VLM_T.required_paths(city, stem)}
            if all(os.path.exists(v) for k, v in e.items() if k.endswith('_path')):
                entries.append(e)
    return entries


def stratified_holdout_split(samples, num_holdout, seed=42):
    """Deterministically hold out `num_holdout` samples from `samples`,
    drawn proportionally from each city so every city subfolder is
    represented (largest-remainder rounding to hit the exact total).

    Returns (remaining_samples, holdout_samples). The proxy disagreement
    rate must be measured on images the target model never trains on this
    run -- otherwise it's partly a train-set metric, not a held-out one.
    """
    by_city = {}
    for s in samples:
        by_city.setdefault(s['city'], []).append(s)

    total = len(samples)
    if num_holdout <= 0 or num_holdout >= total:
        raise ValueError(
            f"num_holdout ({num_holdout}) must be in (0, {total})")

    # Largest-remainder allocation: proportional quota per city, rounded
    # down, then remaining slots go to the cities with the largest
    # fractional remainder so the total matches num_holdout exactly.
    quotas = {city: num_holdout * len(items) / total
              for city, items in by_city.items()}
    counts = {city: int(q) for city, q in quotas.items()}
    remainder = num_holdout - sum(counts.values())
    by_remainder = sorted(quotas.items(), key=lambda kv: kv[1] - int(kv[1]),
                          reverse=True)
    for i in range(remainder):
        counts[by_remainder[i % len(by_remainder)][0]] += 1

    rng = np.random.default_rng(seed)
    remaining, holdout = [], []
    for city in sorted(by_city.keys()):
        items = by_city[city][:]
        rng.shuffle(items)
        n_holdout = min(counts[city], len(items))
        holdout.extend(items[:n_holdout])
        remaining.extend(items[n_holdout:])

    return remaining, holdout


# Unsupervised convergence proxy: fraction of pixels where the target model's
# hardened prediction disagrees with VLM's (frozen) prediction, measured
# on a fixed sample of training images. Only ever touches Pt/Pc-derived
# signal (both already used during training), never Cityscapes ground truth.
def estimate_disagreement_rate(target_model, eval_entries, device):
    target_model.eval()
    total_disagree = 0
    total_valid = 0
    with torch.no_grad():
        for entry in eval_entries:
            img_bgr = cv2.imread(entry['img_path'])
            Pc_label = VLM_T.label(entry, device)
            if img_bgr is None or Pc_label is None:
                continue
            Pc_label = Pc_label.unsqueeze(0)                         # (1,H,W)

            logits = target_model(preprocess(img_bgr).to(device), out_size=(TARGET_H, TARGET_W))
            Pt_label = logits.argmax(dim=1)                          # (1,H,W)

            # VLM pixels of a class absent from the source domain are left out
            valid = (Pc_label < NUM_CLASSES) & ~torch.isin(Pc_label, ABSENT_T)
            disagree = valid & (Pt_label != Pc_label)
            total_disagree += int(disagree.sum().item())
            total_valid += int(valid.sum().item())
    target_model.train()
    freeze_bn(target_model)          # .eval() above flips BN back to train mode too — re-freeze it
    return total_disagree / max(1, total_valid)


def collect_fixed_val_sample(n):
    """A small, deterministic, evenly-spaced sample of Cityscapes val images
    (by filename, not by GT — no labels are loaded), used only for the
    per-epoch qualitative sanity check below."""
    paths = []
    for city in sorted(os.listdir(VAL_IMG_ROOT)):
        city_dir = os.path.join(VAL_IMG_ROOT, city)
        if not os.path.isdir(city_dir):
            continue
        for fname in sorted(os.listdir(city_dir)):
            if fname.endswith(IMG_SUFFIX):
                paths.append((city, fname))
    if not paths or n <= 0:
        return []
    idx = np.linspace(0, len(paths) - 1, num=min(n, len(paths)), dtype=int)
    return [paths[i] for i in idx]


def save_training_viz(target_model, val_sample, device, out_dir, epoch):
    """Diagnostic-only qualitative check, run once per epoch: image | colorized
    prediction panels for a fixed small sample of Cityscapes val images. GT
    labels are never loaded here -- this computes no mIoU and has zero
    influence on training/checkpointing/early-stopping, it exists purely so
    a run can be eyeballed for sanity while it trains."""
    if not val_sample:
        return
    epoch_dir = os.path.join(out_dir, f'epoch{epoch:02d}')
    os.makedirs(epoch_dir, exist_ok=True)
    target_model.eval()
    with torch.no_grad():
        for city, fname in val_sample:
            img = cv2.imread(os.path.join(VAL_IMG_ROOT, city, fname))
            if img is None:
                continue
            pred = predict(target_model, img, device)
            pred_colored = colorize(pred)
            h, w = img.shape[:2]
            panel = np.hstack([
                cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA),
                cv2.resize(pred_colored, (w // 2, h // 2), interpolation=cv2.INTER_AREA),
            ])
            stem = fname[:-len(IMG_SUFFIX)]
            cv2.imwrite(os.path.join(epoch_dir, f'{city}_{stem}.png'), panel)
    target_model.train()
    freeze_bn(target_model)          # .eval() above flips BN back to train mode too — re-freeze it


def entropy(p, dim=-1):
    """Shannon entropy (nats) of a probability distribution along dim."""
    return -(p * torch.log(p + EPS)).sum(dim=dim)


def temperature_soften(p, T, dim=-1):
    """Post-hoc temperature-softening of an already-computed probability
    distribution: raises each probability to the power 1/T and
    renormalizes -- equivalent to softmax(logits/T) for T>1, applied
    directly to p instead of the logits that produced it.

    Used only for q_t's confidence estimate (STEP 5): cross-entropy
    training mechanically sharpens the target model's softmax toward
    one-hot as loss drops, regardless of whether the prediction is
    correct, while VLM is frozen and has no such runaway. Left
    uncorrected, q_t races past q_c within 1-2 epochs purely from CE
    sharpening, so alpha = q_t/(q_t+q_c) > 0.5 almost everywhere and the
    Case-2 "informative disagreement" teacher silently becomes the target
    model's own (increasingly confident but not necessarily correct)
    prediction instead of VLM's -- a second self-reinforcement loop,
    on top of (and independent from) the epoch-1 reference-label fix.
    """
    powered = p.clamp_min(EPS) ** (1.0 / T)
    return powered / powered.sum(dim=dim, keepdim=True)


def compute_boundary_agreement(region_idx, maps, y_s, y_c, ref_key='reference_flat'):
    """STEP 5 (boundary term): fraction of the region's *boundary* pixels
    whose own prediction matches that model's regional majority label
    (y_s / y_c), for each of the two models. Unlike cons_t/cons_c (which
    average over every pixel in the region), this isolates the pixels most
    at risk of straddling the true object edge -- where SAM2 over-/under-
    segmentation is most likely -- so a model that's only self-consistent
    in the interior doesn't score as reliable as one that's also
    consistent right at the mask boundary.

    Always uses the FULL region (region_idx), never the STEP-2 capped
    subsample: boundary detection needs the region's actual contiguous
    shape, and a random subsample of a large region would make almost
    every sampled pixel look like a boundary pixel (its true neighbours
    are very unlikely to also be in the subsample), which would just
    degenerate into (a noisy re-measurement of) cons_t/cons_c.
    """
    H, W = TARGET_H, TARGET_W
    rows = region_idx // W
    cols = region_idx % W

    region_mask = torch.zeros(H, W, dtype=torch.bool, device=region_idx.device)
    region_mask[rows, cols] = True

    up = torch.zeros_like(region_mask)
    down = torch.zeros_like(region_mask)
    left = torch.zeros_like(region_mask)
    right = torch.zeros_like(region_mask)
    up[1:, :] = region_mask[:-1, :]
    down[:-1, :] = region_mask[1:, :]
    left[:, 1:] = region_mask[:, :-1]
    right[:, :-1] = region_mask[:, 1:]

    # A pixel is a boundary pixel if it's IN the region but at least one
    # neighbour is NOT in the region.
    neighbour_outside = region_mask & (~up | ~down | ~left | ~right)
    # Image-edge pixels have no neighbour on the off-frame side -- already
    # implied by the shifts above (that edge's shifted array is all False
    # there), kept explicit here to match the boundary definition exactly.
    neighbour_outside[0, :] |= region_mask[0, :]
    neighbour_outside[-1, :] |= region_mask[-1, :]
    neighbour_outside[:, 0] |= region_mask[:, 0]
    neighbour_outside[:, -1] |= region_mask[:, -1]

    bdry_rows, bdry_cols = torch.where(neighbour_outside)
    n_boundary = len(bdry_rows)
    if n_boundary == 0:
        return 1.0, 1.0   # degenerate/empty region -- treat as perfectly consistent

    bdry_flat = bdry_rows * W + bdry_cols
    ref_bdry = maps[ref_key][bdry_flat]
    pc_bdry = maps['Pc_label_flat'][bdry_flat]
    bdry_t = (ref_bdry == y_s).float().mean().item()
    bdry_c = (pc_bdry == y_c).float().mean().item()
    return bdry_t, bdry_c


def compute_image_maps(target_model, entry, text_embed, device, epoch):
    """Forward pass + VLM maps for one image. Returns the flat per-pixel
    tensors shared by every SAM2 region in this image, or None if any input
    file is unreadable/malformed. The VLM label / probabilities / features
    come from refuse.vlm (built once per image, not per region).

    "reference" (the Ps side of the agreement decision) is epoch-dependent:
    epoch 1 uses the precomputed, frozen source pseudo-label PNG; epoch 2+
    uses the target model's own hardened prediction. Without the epoch-1
    external anchor, Case 1 trains the model to reproduce its own current
    prediction from iteration 1 -- a self-referential loop that collapses
    the model to a single class within the first epoch.
    """
    img_bgr = cv2.imread(entry['img_path'])
    if img_bgr is None:
        return None
    mask_npz = np.load(entry['mask_path'])
    instance_mask_np = mask_npz['instance_mask']
    if instance_mask_np.shape != (TARGET_H, TARGET_W):
        return None

    pc = VLM_T.maps(entry, device)          # frozen VLM: label, probs, L2-normalised features
    if pc is None:
        return None
    Pc_label, Pc_probs_map, fc_norm_map = pc['label'], pc['probs'], pc['feat']
    instance_mask = torch.from_numpy(instance_mask_np.astype(np.int64)).to(device)     # (H,W)

    # ── Forward pass through target model ───────────────────────────────
    logits = target_model(preprocess(img_bgr).to(device), out_size=(TARGET_H, TARGET_W))  # (1,19,H,W)
    Pt_probs = F.softmax(logits, dim=1)

    # The frozen source pseudo-label is loaded in every epoch: it is the
    # epoch-1 reference and the anchor of the VLM-blind class rule (C).
    Ps_np = cv2.imread(entry['ps_path'], cv2.IMREAD_GRAYSCALE)
    if Ps_np is None or Ps_np.shape != (TARGET_H, TARGET_W):
        return None
    ps_label = torch.from_numpy(Ps_np.astype(np.int64)).to(device)               # (H,W), frozen
    if epoch == 1:
        reference_label = ps_label
    else:
        reference_label = Pt_probs.argmax(dim=1)[0]   # (H,W) -- target model's own hardened prediction

    valid = Pc_label < NUM_CLASSES
    return {
        'logits': logits,                                             # (1,19,H,W), keeps grad
        'Pt_probs_flat': Pt_probs[0].reshape(NUM_CLASSES, -1),         # (19,H*W), keeps grad
        'reference_flat': reference_label.reshape(-1),                 # (H*W,)
        'ps_flat': ps_label.reshape(-1),                               # (H*W,), frozen source Ps
        'fc_flat': fc_norm_map.reshape(fc_norm_map.shape[0], -1),      # (512,h*w), L2-normalised
        'fc_lowres_map': lowres_index_map(fc_norm_map.shape[1], fc_norm_map.shape[2], device),
        'Pc_label_flat': Pc_label.reshape(-1),                         # (H*W,)
        'Pc_probs_flat': Pc_probs_map.reshape(NUM_CLASSES, -1),        # (19,H*W), detached
        'instance_mask_flat': instance_mask.reshape(-1),               # (H*W,)
        'valid': valid,
        'valid_flat': valid.reshape(-1),
    }


_lowres_map_cache = {}


def lowres_index_map(h, w, device):
    """(TARGET_H*TARGET_W,) flat index of the nearest (h, w) feature cell for
    every full-resolution pixel, using align_corners=True geometry."""
    key = (h, w, str(device))
    if key not in _lowres_map_cache:
        ys = torch.round(torch.arange(TARGET_H, device=device, dtype=torch.float32)
                         * (h - 1) / (TARGET_H - 1)).long()
        xs = torch.round(torch.arange(TARGET_W, device=device, dtype=torch.float32)
                         * (w - 1) / (TARGET_W - 1)).long()
        _lowres_map_cache[key] = (ys[:, None] * w + xs[None, :]).reshape(-1)
    return _lowres_map_cache[key]


def region_feature(feat_flat, lowres_map, region_idx):
    """Mean feature of a full-resolution region, pooled on the low-res map.
    feat_flat: (C, h*w)."""
    cells, counts = torch.unique(lowres_map[region_idx], return_counts=True)
    weights = counts.to(feat_flat.dtype)
    return feat_flat[:, cells] @ weights / weights.sum()


def sample_region(region_idx, cap, rng, device):
    """STEP 2: the region itself, or a random cap-sized subsample of it."""
    if region_idx.numel() <= cap:
        return region_idx
    sub = torch.from_numpy(rng.choice(region_idx.numel(), cap, replace=False)).to(device)
    return region_idx[sub]


# ── (A) class-balanced Case 1 CE ──────────────────────────────────────────────
def masked_ce(logits_flat, target_flat, class_weights=None):
    """= F.cross_entropy(..., weight=class_weights, ignore_index=IGNORE_LABEL):
    sum(w[y] * nll) / sum(w[y]). None if no target."""
    pix = torch.nonzero(target_flat != IGNORE_LABEL, as_tuple=True)[0]
    if pix.numel() == 0:
        return None
    target = target_flat[pix]
    nll = -F.log_softmax(logits_flat[:, pix], dim=0).gather(0, target[None])[0]
    if class_weights is None:
        return nll.mean()
    w = class_weights[target]
    return (w * nll).sum() / w.sum()


def class_balance_weights(class_freq, power, max_weight):
    """w_c = clamp((mean f over seen classes / f_c) ** power, max=max_weight)
    for classes seen so far (f_c > 0); 1 for the others. power 0 -> all 1."""
    w = torch.ones_like(class_freq)
    seen = class_freq > 0
    if power > 0 and seen.any():
        w[seen] = (class_freq[seen].mean() / class_freq[seen]) ** power
    return w.clamp(max=max_weight)


def update_class_freq(class_freq, target_flat, momentum):
    """In-place EMA of the per-class share of hard-labelled pixels with this
    image's histogram. The first update (all-zero freq) copies it."""
    labels = target_flat[target_flat != IGNORE_LABEL]
    if labels.numel() == 0:
        return
    hist = torch.from_numpy(np.bincount(labels.cpu().numpy(), minlength=NUM_CLASSES)).to(class_freq)
    hist /= hist.sum()
    if class_freq.sum() == 0:
        class_freq.copy_(hist)
    else:
        class_freq.mul_(momentum).add_(hist, alpha=1.0 - momentum)


# ── (B) text z-score verification of Case 1 ───────────────────────────────────
def parse_class_list(spec):
    """'bus,train' -> [15, 16] (Cityscapes class names)."""
    return [CITYSCAPES_CLASSES.index(n.strip()) for n in spec.split(',') if n.strip()]


def parse_verify_pairs(spec):
    """'bus:train,bus:truck' -> {15: [16, 14]}: Case 1 label -> confusers it is checked against."""
    pairs = defaultdict(list)
    for item in spec.split(','):
        if item.strip():
            y, c = (CITYSCAPES_CLASSES.index(n.strip()) for n in item.split(':'))
            pairs[y].append(c)
    return dict(pairs)


def auto_verify_pairs(text_embed, labels, n_pixels, z_classes, topk, max_share):
    """GT-free confuser pairs: every RARE Case 1 label y -- share of the
    warm-start Case 1 pixels (counted over regions with labels in z_classes)
    below max_share -- is checked against its topk nearest classes in the CLIP
    text embedding space (cosine, all 19 classes). Returns
    ({y: [confusers]}, {y: share})."""
    px = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    for y, n in zip(labels, n_pixels):
        if y in z_classes:
            px[y] += n
    share = px / px.sum().clamp_min(1)
    sim = text_embed @ text_embed.T
    sim.fill_diagonal_(float('-inf'))
    if ABSENT_CLASSES:
        sim[:, ABSENT_CLASSES] = float('-inf')        # never a confuser: not in the source domain
    pairs = {}
    for y in z_classes:
        if 0 < share[y] < max_share:
            pairs[y] = sim[y].topk(topk).indices.tolist()
    return pairs, {y: float(share[y]) for y in z_classes if share[y] > 0}


def compute_text_z_stats(fc_regions, labels, z_classes, text_embed):
    """Per text class mean / std of cos(region feature, text_k) over the Case 1
    regions whose label is in z_classes (one vote per region).
    fc_regions: (N, 512) on CPU; labels: list of N ints."""
    keep = torch.tensor([l in z_classes for l in labels], dtype=torch.bool)
    n = int(keep.sum())
    if n < 2:
        raise RuntimeError(f'Only {n} warm-start Case 1 regions in --text-z-classes; cannot '
                           'compute text z-score statistics (use more --warmstart-images).')
    cos = F.normalize(fc_regions[keep].to(text_embed.device), dim=1) @ text_embed.T   # (n, 19)
    return {'mu': cos.mean(dim=0), 'sd': cos.std(dim=0).clamp_min(1e-6), 'n_regions': n}


def text_verify(fc_region, y, text_embed, text_z):
    """Returns the confusers c of y (text_z['pairs'][y]) whose z-scored text
    similarity beats y's by more than text_z['margin'], largest z(c) - z(y)
    first (empty = accept y), and the relabel target: the strongest confuser
    if its gap exceeds text_z['relabel_margin'] and it is not in
    text_z['no_relabel'], else None. fc_region: pooled VLM feature (512,)."""
    if text_z is None:
        return [], None
    confusers = text_z['pairs'].get(y)
    if not confusers:
        return [], None
    z = (text_embed @ F.normalize(fc_region, dim=0) - text_z['mu']) / text_z['sd']
    gaps = [((z[c] - z[y]).item(), c) for c in confusers]
    gaps = [(gap, c) for gap, c in sorted(gaps, reverse=True) if gap > text_z['margin']]
    target = None
    if gaps and gaps[0][0] > text_z['relabel_margin'] and gaps[0][1] not in text_z['no_relabel']:
        target = gaps[0][1]
    return [c for _, c in gaps], target


# ── (C) VLM-blind classes ─────────────────────────────────────────────────────
def blind_class_trust(sample_idx, region_idx, maps, rs, blind_classes, args):
    """Rule (C). Returns (k, q_src, pix) if the frozen source pseudo-label Ps
    votes a VLM-blind class k for this region, VLM votes something else,
    and q_src = (exp(-H_t) + cons_ps + bdry_ps) / 3 > --blind-q-thresh; pix =
    the region's Ps == k pixels (the ones that get the hard label k). None
    otherwise. In epoch 1 the reference IS Ps, so q_src == q_t."""
    if not blind_classes:
        return None
    ps_votes = maps['ps_flat'][sample_idx]
    k = ps_votes.mode().values.item()
    if k not in blind_classes or rs['y_c'] == k:
        return None
    cons_ps = (ps_votes == k).float().mean().item()
    bdry_ps, _ = compute_boundary_agreement(region_idx, maps, k, rs['y_c'], ref_key='ps_flat')
    conf = math.exp(-entropy(temperature_soften(rs['Pt_mean'], args.qt_temperature)).item())
    q_src = (conf + cons_ps + bdry_ps) / 3.0
    if q_src <= args.blind_q_thresh:
        return None
    pix = region_idx[maps['ps_flat'][region_idx] == k]
    if pix.numel() < args.min_region_pixels:
        return None
    return k, q_src, pix


def auto_blind_classes(ps_px, pc_px, max_pc_share, min_ps_share, max_ratio):
    """Classes VLM (almost) never predicts although the source model does:
    VLM label share < max_pc_share AND < max_ratio * source label share,
    and source label share >= min_ps_share."""
    ps_share = ps_px / ps_px.sum().clamp_min(1)
    pc_share = pc_px / pc_px.sum().clamp_min(1)
    blind = [k for k in range(NUM_CLASSES)
             if k not in ABSENT_CLASSES and pc_share[k] < max_pc_share and pc_share[k] < max_ratio * ps_share[k]
             and ps_share[k] >= min_ps_share]
    return blind, ps_share, pc_share


def warmstart_stats(target_model, warm_entries, text_embed, args, device):
    """One no-grad pass over warm_entries with the epoch-1 reference (Ps):
    initial class share for (A), text z-score statistics + pairs for (B),
    the VLM-blind classes for (C). No GT. Returns (class_freq (19,) on
    device, text_z or None, blind_classes list)."""
    print(f'\nWarm-start statistics on {len(warm_entries)} training images...')
    rng = np.random.default_rng(args.seed + 1)
    case1 = []                                    # (y, n_pix, fc_region on CPU)
    blind_cand = []                               # (k, q_src, n_pix) for every class, thresholded later
    ps_px = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    pc_px = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    all_classes = set(range(NUM_CLASSES))
    q_thresh, args.blind_q_thresh = args.blind_q_thresh, float('-inf')   # collect every candidate
    with torch.no_grad():
        for entry in tqdm(warm_entries, desc='warmstart'):
            maps = compute_image_maps(target_model, entry, text_embed, device, epoch=1)
            if maps is None:
                continue
            v = maps['valid_flat']
            ps_v = maps['ps_flat'][v]
            ps_px += torch.from_numpy(np.bincount(ps_v[ps_v < NUM_CLASSES].cpu().numpy(),
                                                  minlength=NUM_CLASSES)).double()
            pc_px += torch.from_numpy(np.bincount(maps['Pc_label_flat'][v].cpu().numpy(),
                                                  minlength=NUM_CLASSES)).double()
            regions, _ = get_region_indices(maps['instance_mask_flat'], v, args.min_region_pixels)
            for region_idx in regions:
                sample_idx = sample_region(region_idx, args.max_region_sample_pixels, rng, device)
                rs = region_stats(sample_idx, maps, region_idx, args.qt_temperature)
                if rs['y_s'] == rs['y_c']:
                    if rs['agreement_strength'] > args.theta_agree:
                        case1.append((rs['y_s'], region_idx.numel(),
                                      region_feature(maps['fc_flat'], maps['fc_lowres_map'], region_idx).cpu()))
                    continue
                trust = blind_class_trust(sample_idx, region_idx, maps, rs, all_classes, args)
                if trust is not None:
                    blind_cand.append((trust[0], trust[1], trust[2].numel()))
    args.blind_q_thresh = q_thresh

    # (C) blind classes
    if args.blind_classes == 'auto':
        blind, ps_share, pc_share = auto_blind_classes(ps_px, pc_px, args.blind_max_pc_share,
                                                       args.blind_min_ps_share, args.blind_max_ratio)
    else:
        blind = [] if args.blind_classes == 'none' else parse_class_list(args.blind_classes)
        _, ps_share, pc_share = auto_blind_classes(ps_px, pc_px, 0.0, 0.0, 0.0)
    print('Label share source / VLM: ' + '  '.join(
        f'{CITYSCAPES_CLASSES[k]}={ps_share[k]:.4f}/{pc_share[k]:.4f}' for k in range(NUM_CLASSES)))
    print(f'VLM-blind classes ({args.blind_classes}): '
          + (', '.join(CITYSCAPES_CLASSES[k] for k in blind) or 'none'))

    # (B) text z-score statistics and pairs
    text_z = None
    labels = [r[0] for r in case1]
    if args.text_verify == 'on' and case1:
        z_classes = set(parse_class_list(args.text_z_classes))
        text_z = compute_text_z_stats(torch.stack([r[2] for r in case1]), labels, z_classes, text_embed)
        if args.text_verify_pairs == 'auto':
            pairs, share = auto_verify_pairs(text_embed.cpu(), labels, [r[1] for r in case1], z_classes,
                                             args.text_verify_topk, args.text_verify_max_share)
            print('Case 1 pixel share within --text-z-classes: ' + '  '.join(
                f'{CITYSCAPES_CLASSES[y]}={s:.4f}' for y, s in sorted(share.items())))
        else:
            pairs = parse_verify_pairs(args.text_verify_pairs)
        text_z['pairs'] = pairs
        text_z['margin'] = args.text_verify_margin
        # relabel only above the stricter margin, never to a class the VLM cannot name
        text_z['relabel_margin'] = (args.text_relabel_margin if args.text_verify_action == 'relabel'
                                    else float('inf'))
        text_z['no_relabel'] = auto_blind_classes(ps_px, pc_px, args.blind_max_pc_share,
                                                  args.blind_min_ps_share, args.blind_max_ratio)[0]
        print(f"Relabel margin {text_z['relabel_margin']}, never relabel to: "
              + (', '.join(CITYSCAPES_CLASSES[k] for k in text_z['no_relabel']) or 'none'))
        print(f"Text verification pairs ({args.text_verify_pairs}, margin {args.text_verify_margin}, "
              f"{text_z['n_regions']} regions): "
              + ('  '.join(f'{CITYSCAPES_CLASSES[y]}->{"/".join(CITYSCAPES_CLASSES[c] for c in cs)}'
                           for y, cs in sorted(pairs.items())) or 'none'))

    # (A) class share of the pixels that will get a hard label
    px = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    n_checked, n_rejected, n_relabelled = defaultdict(int), defaultdict(int), defaultdict(int)
    for y, n_pix, fc_region in case1:
        if text_z is not None and y in text_z['pairs']:
            n_checked[y] += 1
            rejecting, relabel_to = text_verify(fc_region.to(device), y, text_embed, text_z)
            if rejecting:
                n_rejected[y] += 1
                if relabel_to is not None:
                    px[relabel_to] += n_pix
                    n_relabelled[y] += 1
                continue
        px[y] += n_pix
    n_blind = defaultdict(int)
    for k, q_src, n_pix in blind_cand:
        if k in blind and q_src > args.blind_q_thresh:
            px[k] += n_pix
            n_blind[k] += 1
    if text_z is not None:
        print('Warm-start text verification (relabelled/rejected/checked Case 1 regions): ' + '  '.join(
            f'{CITYSCAPES_CLASSES[y]}={n_relabelled[y]}/{n_rejected[y]}/{n_checked[y]}'
            for y in sorted(text_z['pairs'])))
    if blind:
        print(f'Warm-start blind-class regions trusted (q_src > {args.blind_q_thresh}): ' + '  '.join(
            f'{CITYSCAPES_CLASSES[k]}={n_blind[k]}' for k in blind))
    class_freq = (px / px.sum().clamp_min(1)).float().to(device)
    print('Hard-label class share: ' + '  '.join(
        f'{n}={f:.4f}' for n, f in zip(CITYSCAPES_CLASSES, class_freq.tolist()) if f > 0))
    print('Initial class-balance weights: ' + '  '.join(
        f'{n}={w:.2f}' for n, w, f in zip(
            CITYSCAPES_CLASSES,
            class_balance_weights(class_freq, args.cb_power, args.cb_max_weight).tolist(),
            class_freq.tolist()) if f > 0))
    return class_freq, text_z, blind


def get_region_indices(instance_mask_flat, valid_flat, min_region_pixels):
    """STEP 1: unique SAM2 region ids among valid pixels, dropping regions
    with fewer than min_region_pixels valid pixels. Returns
    (list_of_flat_index_tensors, n_skipped_small)."""
    region_ids = torch.unique(instance_mask_flat[valid_flat])
    regions = []
    n_skipped = 0
    for rid in region_ids.tolist():
        region_idx = torch.nonzero((instance_mask_flat == rid) & valid_flat, as_tuple=True)[0]
        if region_idx.numel() < min_region_pixels:
            n_skipped += 1
            continue
        regions.append(region_idx)
    return regions, n_skipped


def region_stats(sample_idx, maps, region_idx, qt_temperature):
    """STEPS 2-5 for one region, evaluated on sample_idx (the full region,
    or a capped random subsample of it for very large regions -- see
    train_one_epoch) -- except the boundary-agreement term, which always
    uses the full region_idx (see compute_boundary_agreement). Returns
    y_s, y_c, agreement_strength, q_t, q_c, cons_t, cons_c, bdry_t,
    bdry_c, and the two mean probability vectors (needed again in STEP 7
    for the fused/sharpened teacher, if this region turns out to be
    Case 2).

    q_t's entropy term is computed on a temperature-softened copy of
    Pt_mean (qt_temperature > 1 softens) -- see temperature_soften()'s
    docstring for why this is needed."""
    reference_votes = maps['reference_flat'][sample_idx]
    Pc_votes = maps['Pc_label_flat'][sample_idx]

    with torch.no_grad():
        Pt_mean = maps['Pt_probs_flat'][:, sample_idx].mean(dim=1)     # (19,), detached
    Pc_mean = maps['Pc_probs_flat'][:, sample_idx].mean(dim=1)         # (19,), already detached

    y_s = reference_votes.mode().values.item()
    y_c = Pc_votes.mode().values.item()
    agreement_strength = (reference_votes == Pc_votes).float().mean().item()

    # Pixel consensus: fraction of the region's pixels whose own prediction
    # matches that model's regional majority (internal self-consistency).
    cons_t = (reference_votes == y_s).float().mean().item()
    cons_c = (Pc_votes == y_c).float().mean().item()

    # Boundary agreement: same idea, restricted to the region's boundary
    # pixels, computed on the full (uncapped) region.
    bdry_t, bdry_c = compute_boundary_agreement(region_idx, maps, y_s, y_c)

    q_t = (math.exp(-entropy(temperature_soften(Pt_mean, qt_temperature)).item()) + cons_t + bdry_t) / 3.0
    q_c = (math.exp(-entropy(Pc_mean).item()) + cons_c + bdry_c) / 3.0

    return {'y_s': y_s, 'y_c': y_c, 'agreement_strength': agreement_strength,
            'q_t': q_t, 'q_c': q_c, 'cons_t': cons_t, 'cons_c': cons_c,
            'bdry_t': bdry_t, 'bdry_c': bdry_c,
            'Pt_mean': Pt_mean, 'Pc_mean': Pc_mean}


def calibrate_thresholds(target_model, calib_entries, text_embed, args, device):
    """STEP 6: run STEPS 1-5 over a fixed sample of training images (no
    grad -- this never updates the model), pool agreement_strength / q_t /
    q_c / |q_t - q_c| across every region seen, and set thresholds as
    percentiles of the pooled values."""
    print(f'\nCalibrating thresholds on {len(calib_entries)} training images...')
    rng = np.random.default_rng(args.seed)
    agree_vals, max_q_vals, margin_vals = [], [], []
    with torch.no_grad():
        for entry in tqdm(calib_entries, desc='calibration'):
            maps = compute_image_maps(target_model, entry, text_embed, device, epoch=1)
            if maps is None:
                continue
            regions, _ = get_region_indices(maps['instance_mask_flat'], maps['valid_flat'],
                                             args.min_region_pixels)
            for region_idx in regions:
                if region_idx.numel() > args.max_region_sample_pixels:
                    sub = rng.choice(region_idx.numel(), args.max_region_sample_pixels, replace=False)
                    sample_idx = region_idx[torch.from_numpy(sub).to(device)]
                else:
                    sample_idx = region_idx
                rs = region_stats(sample_idx, maps, region_idx, args.qt_temperature)
                agree_vals.append(rs['agreement_strength'])
                max_q_vals.append(max(rs['q_t'], rs['q_c']))
                margin_vals.append(abs(rs['q_t'] - rs['q_c']))

    theta_agree = float(np.percentile(agree_vals, 60))
    theta_rel = float(np.percentile(max_q_vals, 30))
    theta_margin = float(np.percentile(margin_vals, 25))

    print(f'Calibration done on {len(agree_vals)} regions:')
    print(f'  theta_agree  (60th pct of agreement_strength) = {theta_agree:.4f}')
    print(f'  theta_rel    (30th pct of max(q_t,q_c))       = {theta_rel:.4f}')
    print(f'  theta_margin (25th pct of |q_t-q_c|)          = {theta_margin:.4f}')

    return {
        'theta_agree': theta_agree, 'theta_rel': theta_rel, 'theta_margin': theta_margin,
        'n_calibration_images': len(calib_entries), 'n_calibration_regions': len(agree_vals),
    }


# Core training loop for one epoch. For each image, resolves every SAM2
# region (min size --min-region-pixels) into one of 3 cases (clean
# agreement / informative disagreement / ignore) using the calibrated
# thresholds in args.theta_agree/theta_rel/theta_margin, and accumulates
# the corresponding loss terms.
def train_one_epoch(epoch, target_model, optimizer, text_embed, entries, args, device, writer=None,
                    class_freq=None, text_z=None, blind_classes=()):
    rng = np.random.default_rng(args.seed + epoch)
    order = rng.permutation(len(entries))

    stats = {'ce_agree': [], 'ce_disagree': [], 'balance': [], 'total': [],
             'disagree_rate': [], 'n_case1_regions': [], 'n_case2_regions': [],
             'n_case3_regions': [], 'n_skipped_small_regions': [], 'n_case2_px': [],
             'mean_cons_t': [], 'mean_cons_c': [], 'mean_bdry_t': [], 'mean_bdry_c': [],
             'partial': [], 'ce_agree_unweighted': [], 'n_case1_partial': [], 'px_case1_partial': [],
             'n_case1_relabel': [], 'px_case1_relabel': [], 'relabel_pairs': defaultdict(int),
             'n_blind_regions': [], 'px_blind': []}

    pbar = tqdm(order, desc=f'epoch{epoch:02d}')
    for i, idx in enumerate(pbar):
        global_step = (epoch - 1) * len(entries) + i
        entry = entries[idx]

        maps = compute_image_maps(target_model, entry, text_embed, device, epoch=epoch)
        if maps is None:
            continue

        regions, n_skipped_small = get_region_indices(
            maps['instance_mask_flat'], maps['valid_flat'], args.min_region_pixels)

        ce_target_flat = torch.full_like(maps['reference_flat'], IGNORE_LABEL)  # Case-1 hard labels
        used_mask_flat = torch.zeros_like(maps['valid_flat'])                   # Case-1 + Case-2 pixels

        disagree_terms, partial_terms = [], []
        n_case1, n_case2, n_case3, n_case2_px = 0, 0, 0, 0
        n_case1_partial = px_case1_partial = n_case1_relabel = px_case1_relabel = n_blind = px_blind = 0
        region_cons_t, region_cons_c = [], []
        region_bdry_t, region_bdry_c = [], []

        for region_idx in regions:
            # STEP 2: cap the STATS computation for very large regions --
            # the loss below (ce_target_flat / used_mask_flat / disagree_terms)
            # always uses the FULL region_idx, every pixel gets supervised.
            sample_idx = sample_region(region_idx, args.max_region_sample_pixels, rng, device)

            rs = region_stats(sample_idx, maps, region_idx, args.qt_temperature)
            y_s, y_c = rs['y_s'], rs['y_c']
            agreement_strength = rs['agreement_strength']
            q_t, q_c = rs['q_t'], rs['q_c']
            region_cons_t.append(rs['cons_t'])
            region_cons_c.append(rs['cons_c'])
            region_bdry_t.append(rs['bdry_t'])
            region_bdry_c.append(rs['bdry_c'])

            # ── (C) VLM-blind class: the source pseudo-label is the only
            # witness, so it is checked before the Case 1/2 decision.
            trust = blind_class_trust(sample_idx, region_idx, maps, rs, blind_classes, args)
            if trust is not None:
                k, _, pix = trust
                ce_target_flat[pix] = k
                used_mask_flat[pix] = True
                n_blind += 1
                px_blind += pix.numel()
                continue

            if y_s == y_c:
                if agreement_strength > args.theta_agree:
                    # ── CASE 1 — clean agreement ────────────────────────────
                    rejecting, relabel_to = [], None
                    if text_z is not None and y_s in text_z['pairs']:
                        with torch.no_grad():
                            fc_region = region_feature(maps['fc_flat'], maps['fc_lowres_map'], region_idx)
                        rejecting, relabel_to = text_verify(fc_region, y_s, text_embed, text_z)
                    if rejecting and relabel_to is not None:
                        # (B) both models agree on y_s but the CLIP text clearly prefers
                        # a confuser: the strongest confuser becomes the hard label.
                        ce_target_flat[region_idx] = relabel_to
                        used_mask_flat[region_idx] = True
                        n_case1_relabel += 1
                        px_case1_relabel += region_idx.numel()
                        stats['relabel_pairs'][(y_s, relabel_to)] += region_idx.numel()
                        continue
                    if rejecting:
                        # (B) partial label {y_s} + confusers, no hard label.
                        cand = torch.tensor([y_s] + rejecting, device=device)
                        Pt_pix = maps['Pt_probs_flat'][:, region_idx]            # (19,N), attached
                        partial_terms.append(-torch.log(Pt_pix[cand].sum(dim=0) + EPS))
                        used_mask_flat[region_idx] = True
                        n_case1_partial += 1
                        px_case1_partial += region_idx.numel()
                        continue
                    ce_target_flat[region_idx] = y_s
                    used_mask_flat[region_idx] = True
                    n_case1 += 1
                else:
                    # ── CASE 3 — agreed but weak/ambiguous majority ─────────
                    n_case3 += 1
                continue

            if max(q_t, q_c) > args.theta_rel and abs(q_t - q_c) > args.theta_margin:
                # ── CASE 2 — informative disagreement ───────────────────────
                n_case2 += 1
                n_case2_px += region_idx.numel()
                used_mask_flat[region_idx] = True

                alpha = q_t / (q_t + q_c + EPS)
                teacher_probs = rs['Pt_mean'] if alpha > 0.5 else rs['Pc_mean']  # (19,), detached
                log_teacher = torch.log(teacher_probs + EPS)
                P_hat = F.softmax(log_teacher / args.sharpen_temp, dim=-1).detach()  # (19,), sharpened

                Pt_probs_region_pixels = maps['Pt_probs_flat'][:, region_idx].transpose(0, 1)  # (Nfull,19), attached
                region_term = -(P_hat.unsqueeze(0) * torch.log(Pt_probs_region_pixels + EPS)).sum(dim=-1)
                disagree_terms.append(region_term)
            else:
                # ── CASE 3 — disagreement region, not reliable/distinguishable
                n_case3 += 1

        if i < 5:  # only first 5 images per epoch
            total_regions = n_case1 + n_case2 + n_case3
            total_pixels = int(maps['valid_flat'].sum().item())
            case1_pixels = int((ce_target_flat != IGNORE_LABEL).sum().item())
            print(f"\n[diag epoch{epoch} img{i}]"
                  f" regions: case1={n_case1} case1_partial={n_case1_partial}"
                  f" case1_relabel={n_case1_relabel} blind={n_blind}"
                  f" case2={n_case2} case3={n_case3}"
                  f" skipped_small={n_skipped_small} total_considered={total_regions}"
                  f" pixels: case1={case1_pixels} case2={n_case2_px} valid={total_pixels}"
                  f" case1_frac={case1_pixels/max(1,total_pixels):.3f}"
                  f" case2_frac={n_case2_px/max(1,total_pixels):.3f}"
                  f" case3_frac={(total_pixels-case1_pixels-n_case2_px)/max(1,total_pixels):.3f}")

        # ── L_CE_agree — hard-label CE on the Case-1 (+ blind-class) pixels,
        # (A) class-balanced: weights from the class share BEFORE this image,
        # then the share is updated with this image's hard labels.
        zero = torch.tensor(0., device=device)
        logits_flat = maps['logits'][0].reshape(NUM_CLASSES, -1)
        cb_weights = class_balance_weights(class_freq, args.cb_power, args.cb_max_weight)
        L_CE_agree = masked_ce(logits_flat, ce_target_flat, cb_weights)
        L_CE_agree = zero if L_CE_agree is None else L_CE_agree
        with torch.no_grad():
            L_CE_agree_unw = masked_ce(logits_flat, ce_target_flat)              # logged only
            L_CE_agree_unw = zero if L_CE_agree_unw is None else L_CE_agree_unw
        update_class_freq(class_freq, ce_target_flat, args.cb_momentum)

        # ── L_partial — (B) Case-1 regions rejected by the text check ──────
        L_partial = torch.cat(partial_terms).mean() if partial_terms else zero

        # ── L_CE_disagree — from all Case-2 regions ────────────────────────
        if disagree_terms:
            L_CE_disagree = torch.cat(disagree_terms).mean()
        else:
            L_CE_disagree = torch.tensor(0., device=device)

        # ── L_balance — class-marginal entropy regularizer, Case-1+2 pixels only
        if used_mask_flat.any():
            mean_probs = maps['Pt_probs_flat'][:, used_mask_flat].mean(dim=1)
            L_balance = -entropy(mean_probs)
        else:
            L_balance = torch.tensor(0., device=device)

        L_total = (L_CE_agree + args.lambda_disagree * L_CE_disagree
                   + args.lambda_partial * L_partial + args.mu_balance * L_balance)

        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()

        n_valid = int(maps['valid'].sum().item())
        mean_cons_t_img = float(np.mean(region_cons_t)) if region_cons_t else float('nan')
        mean_cons_c_img = float(np.mean(region_cons_c)) if region_cons_c else float('nan')
        mean_bdry_t_img = float(np.mean(region_bdry_t)) if region_bdry_t else float('nan')
        mean_bdry_c_img = float(np.mean(region_bdry_c)) if region_bdry_c else float('nan')

        stats['ce_agree'].append(L_CE_agree.item())
        stats['ce_disagree'].append(L_CE_disagree.item())
        stats['balance'].append(L_balance.item())
        stats['total'].append(L_total.item())
        stats['n_case1_regions'].append(n_case1)
        stats['n_case2_regions'].append(n_case2)
        stats['n_case3_regions'].append(n_case3)
        stats['n_skipped_small_regions'].append(n_skipped_small)
        stats['n_case2_px'].append(n_case2_px)
        stats['disagree_rate'].append(n_case2_px / max(1, n_valid))
        stats['mean_cons_t'].append(mean_cons_t_img)
        stats['mean_cons_c'].append(mean_cons_c_img)
        stats['mean_bdry_t'].append(mean_bdry_t_img)
        stats['mean_bdry_c'].append(mean_bdry_c_img)
        stats['partial'].append(L_partial.item())
        stats['ce_agree_unweighted'].append(L_CE_agree_unw.item())
        stats['n_case1_partial'].append(n_case1_partial)
        stats['px_case1_partial'].append(px_case1_partial)
        stats['n_case1_relabel'].append(n_case1_relabel)
        stats['px_case1_relabel'].append(px_case1_relabel)
        stats['n_blind_regions'].append(n_blind)
        stats['px_blind'].append(px_blind)

        if writer is not None:
            writer.add_scalar('train_iter/L_CE_agree', L_CE_agree.item(), global_step)
            writer.add_scalar('train_iter/L_CE_disagree', L_CE_disagree.item(), global_step)
            writer.add_scalar('train_iter/L_balance', L_balance.item(), global_step)
            writer.add_scalar('train_iter/L_total', L_total.item(), global_step)
            writer.add_scalar('train_iter/n_case1_regions', n_case1, global_step)
            writer.add_scalar('train_iter/n_case2_regions', n_case2, global_step)
            writer.add_scalar('train_iter/n_case3_regions', n_case3, global_step)
            writer.add_scalar('train_iter/n_skipped_small_regions', n_skipped_small, global_step)
            writer.add_scalar('train_iter/mean_cons_t', mean_cons_t_img, global_step)
            writer.add_scalar('train_iter/mean_cons_c', mean_cons_c_img, global_step)
            writer.add_scalar('train_iter/mean_bdry_t', mean_bdry_t_img, global_step)
            writer.add_scalar('train_iter/mean_bdry_c', mean_bdry_c_img, global_step)
            writer.add_scalar('train_iter/L_partial', L_partial.item(), global_step)
            writer.add_scalar('train_iter/L_CE_agree_unweighted', L_CE_agree_unw.item(), global_step)
            writer.add_scalar('train_iter/n_case1_partial', n_case1_partial, global_step)
            writer.add_scalar('train_iter/px_case1_relabel', px_case1_relabel, global_step)
            writer.add_scalar('train_iter/px_blind', px_blind, global_step)

        pbar.set_postfix(agree=f"{L_CE_agree.item():.3f}",
                          dis=f"{L_CE_disagree.item():.3f}",
                          bal=f"{L_balance.item():.3f}")

        if len(stats['total']) % 50 == 0:
            torch.cuda.empty_cache()

    return stats


def main():
    parser = argparse.ArgumentParser(description='ReFuSe Stage 1')
    parser.add_argument('--config', required=True, help='Experiment config (configs/*.yaml)')
    parser.add_argument('--work-dir', default=None, help='Overrides work_dir of the config')
    parser.add_argument('--start-epoch', type=int, default=1,
                        help='Epoch to begin at (1-indexed). Epoch 1 starts from the source model '
                             '(source_ckpt of the config); epoch N>1 resumes from dsp_epoch<N-1>.pth.')
    parser.add_argument('--max-epochs', type=int, default=30,
                        help='Upper bound on adaptation epochs — training may stop earlier '
                             'via the proxy-metric early stop (see --early-stop-patience)')
    parser.add_argument('--disagree-eval-images', type=int, default=200,
                        help='Number of training images sampled once (fixed across epochs, seeded '
                             'by --seed) to estimate the target-vs-VLM disagreement rate proxy')
    parser.add_argument('--early-stop-patience', type=int, default=5,
                        help='Stop once the proxy disagreement rate has not reached a new low for '
                             'this many consecutive epochs (0 = disabled, always run to --max-epochs)')
    parser.add_argument('--min-region-pixels', type=int, default=50,
                        help='Skip SAM2 regions with fewer than this many valid pixels (STEP 1)')
    parser.add_argument('--calibration-images', type=int, default=50,
                        help='Number of training images used to calibrate theta_agree/theta_rel/'
                             'theta_margin once before training starts (STEP 6). Cached in '
                             'thresholds.json; skipped on later runs unless --force-recalibrate.')
    parser.add_argument('--force-recalibrate', action='store_true',
                        help='Recompute thresholds.json even if it already exists')
    parser.add_argument('--qt-temperature', type=float, default=2.0,
                        help='Temperature T>1 used to soften Pt_mean before computing q_t\'s '
                             'entropy term (STEP 5) -- corrects for CE training mechanically '
                             'sharpening the target model\'s confidence toward one-hot regardless '
                             'of correctness, which otherwise lets q_t race past q_c (VLM\'s, '
                             'frozen/uncorrected) within 1-2 epochs and turns Case 2 into a second '
                             'self-reinforcement loop. See temperature_soften()\'s docstring. '
                             'T=1 disables softening (raw entropy, the pre-fix behavior).')
    parser.add_argument('--sharpen-temp', type=float, default=0.5,
                        help='Temperature tau for sharpening the Case-2 teacher distribution '
                             '(P_hat = softmax(log(teacher_probs)/tau)); tau<1 sharpens')
    parser.add_argument('--lambda-disagree', type=float, default=1.0,
                        help='Weight lambda on L_CE_disagree in L_total')
    parser.add_argument('--mu-balance', type=float, default=0.01,
                        help='Weight mu on L_balance (class-marginal entropy regularizer) in L_total')
    parser.add_argument('--lr', type=float, default=None,
                        help='Adam learning rate (default: 1e-5 for deeplabv2, 6e-5 for mitb5)')
    parser.add_argument('--max-region-sample-pixels', type=int, default=5000,
                        help='Cap pixels used per region when computing votes/means/reliability '
                             '(STEPS 2-5) for very large regions — the region-level decision '
                             'itself, and the CE loss, still cover every pixel in the region.')
    parser.add_argument('--ckpt-interval', type=int, default=5,
                        help='Save a full dsp_epoch<N>.pth checkpoint only at epoch 1 and every '
                             'N-th epoch thereafter (e.g. 1, 5, 10, 15, ...), to avoid filling disk '
                             'with every intermediate epoch. best.pth is still saved every time a '
                             'new best proxy disagreement rate is reached, regardless of this '
                             'interval. NOTE: --start-epoch can only resume from an epoch that was '
                             'actually saved under this interval.')
    parser.add_argument('--viz-count', type=int, default=10,
                        help='Number of Cityscapes val images (evenly spaced, image-only -- no GT '
                             'is loaded) to render image/prediction sanity-check panels for after '
                             'every epoch, saved under train_qualitative_viz/epoch<N>/. This is '
                             'diagnostic only: no labels are consulted and it has no effect on '
                             'training, checkpointing, or early stopping. Set to 0 to disable.')
    parser.add_argument('--run-name', default='stage1',
                        help='Checkpoint directory name under checkpoints/. One per ablation.')
    parser.add_argument('--warmstart-images', type=int, default=500,
                        help='Training images of the no-grad warm-start pass (class share, text '
                             'z-score statistics and pairs, VLM-blind classes)')
    # (A) class-balanced Case 1 CE
    parser.add_argument('--cb-power', type=float, default=0.5,
                        help='w_c = (mean f / f_c) ** cb_power on L_CE_agree; 0 = unweighted')
    parser.add_argument('--cb-max-weight', type=float, default=3.0)
    parser.add_argument('--cb-momentum', type=float, default=0.99,
                        help='Per-image EMA momentum of the hard-label class share')
    # (B) text z-score verification of Case 1
    parser.add_argument('--text-verify', choices=['on', 'off'], default='on')
    parser.add_argument('--text-verify-pairs', default='auto',
                        help="'auto' (rare Case 1 labels vs their nearest text classes) or an "
                             "explicit list such as 'bus:train,bus:truck'")
    parser.add_argument('--text-verify-action', choices=['relabel', 'partial'], default='relabel',
                        help='A rejected Case 1 region gets the strongest confuser as its hard '
                             "label if z(c) - z(y) > --text-relabel-margin ('relabel'), or always "
                             "the partial label {y, confusers} ('partial')")
    parser.add_argument('--text-relabel-margin', type=float, default=1.0,
                        help='z(c) - z(y) above which a rejected region is hard-labelled c; between '
                             '--text-verify-margin and this it gets the partial label')
    parser.add_argument('--text-verify-topk', type=int, default=2)
    parser.add_argument('--text-verify-max-share', type=float, default=0.01)
    parser.add_argument('--text-verify-margin', type=float, default=0.5)
    parser.add_argument('--text-z-classes', default=None,
                        help='Case 1 labels whose regions define the z-score statistics (default: building, wall, fence, '
                             'person, rider, car, truck, bus, train, motorcycle, bicycle minus the classes '
                             'absent from the source domain)')
    parser.add_argument('--lambda-partial', type=float, default=1.0,
                        help='Weight of L_partial (Case 1 regions rejected by the text check)')
    # (C) VLM-blind classes
    parser.add_argument('--blind-classes', default='auto',
                        help="'auto', 'none', or class names such as 'terrain': classes VLM "
                             'cannot name, labelled from the frozen source pseudo-label')
    parser.add_argument('--blind-max-pc-share', type=float, default=5e-4,
                        help="auto: a class is blind if its share of VLM's labels is below this")
    parser.add_argument('--blind-max-ratio', type=float, default=0.05,
                        help="auto: ... and below this fraction of its share of the source pseudo-labels")
    parser.add_argument('--blind-min-ps-share', type=float, default=1e-3,
                        help='auto: ... and its share of the source pseudo-labels is at least this')
    parser.add_argument('--blind-q-thresh', type=float, default=0.6,
                        help='q_src above which a blind-class region gets its hard label')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true',
                        help='Process only the first 2 train images per city (smoke test)')
    args = parser.parse_args()

    global BB, VLM_T, ABSENT_CLASSES, ABSENT_T, VAL_IMG_ROOT
    cfg = load_config(args.config)
    BB = make_backbone(cfg)
    ABSENT_CLASSES = cfg['absent_classes']
    VAL_IMG_ROOT = cfg['data']['val_img_dir']
    if args.lr is None:
        args.lr = BB.default_lr
    if args.text_z_classes is None:
        args.text_z_classes = default_text_z_classes(ABSENT_CLASSES)
    args.work_dir = args.work_dir or cfg['work_dir']
    print(f"Config {cfg['config_path']}: source={cfg['source']} backbone={BB.name} vlm={cfg['vlm']['name']} "
          f"lr={args.lr}")
    CKPT_DIR = os.path.join(args.work_dir, args.run_name)
    TRAIN_LOG_PATH = f'{CKPT_DIR}/dsp_train_log.jsonl'
    TB_LOG_DIR = f'{CKPT_DIR}/tensorboard'
    THRESHOLDS_PATH = f'{CKPT_DIR}/thresholds.json'
    WARMSTART_PATH = f'{CKPT_DIR}/warmstart_stats.json'
    TRAIN_VIZ_DIR = f'{CKPT_DIR}/train_qualitative_viz'
    print(f'Run: {args.run_name} -> {CKPT_DIR}')

    device = torch.device(f'cuda:{args.gpu}')
    torch.manual_seed(args.seed)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=TB_LOG_DIR)

    ABSENT_T = torch.tensor(ABSENT_CLASSES, dtype=torch.long, device=device)
    VLM_T = VLM.create(cfg['vlm'], device, ABSENT_CLASSES)

    # ── Bootstrap or resume ────────────────────────────────────────────────
    if args.start_epoch == 1:
        target_state = None
        optimizer_state = None
        ckpt = None
        print(f"Bootstrapping from the source model {cfg['source_ckpt']}")
    else:
        resume_path = os.path.join(CKPT_DIR, f'dsp_epoch{args.start_epoch - 1}.pth')
        ckpt = torch.load(resume_path, map_location='cpu')
        target_state = ckpt['target_model']
        optimizer_state = ckpt['optimizer']
        print(f'Resuming from {resume_path}  (epoch {ckpt["epoch"]})')

    target_model = BB.build(device)                       # source-only weights
    if target_state is not None:
        target_model.load_state_dict(target_state, strict=True)
    target_model.train()
    freeze_bn(target_model)
    for p in target_model.parameters():
        p.requires_grad = True

    text_embed = VLM_T.text_embed                         # (19, 512), L2-normalised

    optimizer = torch.optim.Adam(target_model.parameters(), lr=args.lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    all_entries = collect_entries(cfg['data']['ps_root'], cfg['data']['train_img_dir'],
                                  cfg['data']['sam2_root'])
    if args.debug:
        seen = {}
        all_entries = [e for e in all_entries if seen.get(e['city'], 0) < 2 and not seen.__setitem__(e['city'], seen.get(e['city'], 0) + 1)]

    # Stratified (per-city) train/val holdout: the disagreement-rate proxy
    # is measured on images the target model never trains on this run,
    # instead of a random subsample of the same pool it was just optimized
    # on. Capped at half the pool so --debug (few images/city) still leaves
    # a usable training set.
    n_eval = min(args.disagree_eval_images, max(1, len(all_entries) // 2))
    entries, eval_entries = stratified_holdout_split(all_entries, n_eval, seed=args.seed)
    print(f'{len(entries)} train images  ({len(eval_entries)} held out for the disagreement-rate proxy)')

    # ── STEP 6: dynamic threshold calibration (cached) ─────────────────────
    if os.path.exists(THRESHOLDS_PATH) and not args.force_recalibrate:
        with open(THRESHOLDS_PATH) as f:
            thresholds = json.load(f)
        print(f'Loaded cached thresholds from {THRESHOLDS_PATH}: {thresholds}')
    else:
        calib_rng = np.random.default_rng(args.seed)
        n_calib = min(args.calibration_images, len(entries))
        calib_idx = calib_rng.choice(len(entries), n_calib, replace=False)
        calib_entries = [entries[i] for i in calib_idx]
        thresholds = calibrate_thresholds(target_model, calib_entries, text_embed, args, device)
        with open(THRESHOLDS_PATH, 'w') as f:
            json.dump(thresholds, f, indent=2)
        print(f'Saved thresholds -> {THRESHOLDS_PATH}')

    args.theta_agree = thresholds['theta_agree']
    args.theta_rel = thresholds['theta_rel']
    args.theta_margin = thresholds['theta_margin']

    # ── Warm-start statistics for (A)/(B)/(C); restored from the checkpoint on resume
    if ckpt is not None:
        class_freq = ckpt['class_freq'].to(device)
        text_z = ckpt['text_z']
        if text_z is not None:
            text_z = {k: v.to(device) if torch.is_tensor(v) else v for k, v in text_z.items()}
        blind_classes = ckpt['blind_classes']
        print('Restored from checkpoint: blind classes = '
              + (', '.join(CITYSCAPES_CLASSES[k] for k in blind_classes) or 'none')
              + ', text pairs = ' + (str({CITYSCAPES_CLASSES[y]: [CITYSCAPES_CLASSES[c] for c in cs]
                                          for y, cs in text_z['pairs'].items()}) if text_z else 'off'))
    else:
        warm_rng = np.random.default_rng(args.seed + 1)
        n_warm = min(args.warmstart_images, len(entries))
        warm_entries = [entries[i] for i in sorted(warm_rng.choice(len(entries), n_warm, replace=False))]
        class_freq, text_z, blind_classes = warmstart_stats(target_model, warm_entries, text_embed,
                                                            args, device)
        with open(WARMSTART_PATH, 'w') as f:
            json.dump({
                'n_images': n_warm,
                'blind_classes': [CITYSCAPES_CLASSES[k] for k in blind_classes],
                'class_share': dict(zip(CITYSCAPES_CLASSES, class_freq.tolist())),
                'class_weights': dict(zip(CITYSCAPES_CLASSES, class_balance_weights(
                    class_freq, args.cb_power, args.cb_max_weight).tolist())),
                'text_z': None if text_z is None else {
                    'n_regions': text_z['n_regions'], 'margin': text_z['margin'],
                    'relabel_margin': text_z['relabel_margin'],
                    'no_relabel': [CITYSCAPES_CLASSES[k] for k in text_z['no_relabel']],
                    'pairs': {CITYSCAPES_CLASSES[y]: [CITYSCAPES_CLASSES[c] for c in cs]
                              for y, cs in sorted(text_z['pairs'].items())},
                    'mu': dict(zip(CITYSCAPES_CLASSES, text_z['mu'].tolist())),
                    'sd': dict(zip(CITYSCAPES_CLASSES, text_z['sd'].tolist()))},
                'args': {k: v for k, v in vars(args).items()},
            }, f, indent=2)
        print(f'Saved warm-start statistics -> {WARMSTART_PATH}')
    blind_classes = set(blind_classes)

    val_sample = collect_fixed_val_sample(args.viz_count)
    if val_sample:
        print(f'Qualitative sanity-check viz: {len(val_sample)} Cityscapes val images '
              f'-> {TRAIN_VIZ_DIR}/epoch<N>/ (diagnostic only, no GT loaded)')

    # Early-stopping state is saved in every checkpoint, so a resume neither
    # resets the patience counter nor overwrites best.pth with a worse epoch.
    best_disagree_rate = ckpt['best_disagree_rate'] if ckpt is not None else float('inf')
    epochs_no_improve = ckpt['epochs_no_improve'] if ckpt is not None else 0

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        stats = train_one_epoch(epoch, target_model, optimizer, text_embed, entries, args, device,
                                writer=writer, class_freq=class_freq, text_z=text_z,
                                blind_classes=blind_classes)

        mean_ce_agree    = float(np.mean(stats['ce_agree'])) if stats['total'] else float('nan')
        mean_ce_disagree = float(np.mean(stats['ce_disagree'])) if stats['total'] else float('nan')
        mean_balance     = float(np.mean(stats['balance'])) if stats['total'] else float('nan')
        mean_cons_t      = float(np.nanmean(stats['mean_cons_t'])) if stats['mean_cons_t'] else float('nan')
        mean_cons_c      = float(np.nanmean(stats['mean_cons_c'])) if stats['mean_cons_c'] else float('nan')
        mean_bdry_t      = float(np.nanmean(stats['mean_bdry_t'])) if stats['mean_bdry_t'] else float('nan')
        mean_bdry_c      = float(np.nanmean(stats['mean_bdry_c'])) if stats['mean_bdry_c'] else float('nan')
        total_case1 = int(sum(stats['n_case1_regions']))
        total_case2 = int(sum(stats['n_case2_regions']))
        total_case3 = int(sum(stats['n_case3_regions']))
        total_skipped_small = int(sum(stats['n_skipped_small_regions']))
        total_case2_px = int(sum(stats['n_case2_px']))
        mean_partial     = float(np.mean(stats['partial'])) if stats['total'] else float('nan')
        mean_ce_agree_unw = float(np.mean(stats['ce_agree_unweighted'])) if stats['total'] else float('nan')
        total_case1_partial = int(sum(stats['n_case1_partial']))
        total_case1_partial_px = int(sum(stats['px_case1_partial']))
        total_case1_relabel = int(sum(stats['n_case1_relabel']))
        total_case1_relabel_px = int(sum(stats['px_case1_relabel']))
        relabel_pairs = {f'{CITYSCAPES_CLASSES[y]}->{CITYSCAPES_CLASSES[c]}': n
                         for (y, c), n in sorted(stats['relabel_pairs'].items())}
        total_blind = int(sum(stats['n_blind_regions']))
        total_blind_px = int(sum(stats['px_blind']))
        cb_w = class_balance_weights(class_freq, args.cb_power, args.cb_max_weight)

        disagree_rate_eval = estimate_disagreement_rate(target_model, eval_entries, device)
        is_best = disagree_rate_eval < best_disagree_rate - 1e-6
        if is_best:
            best_disagree_rate = disagree_rate_eval
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f'\n[epoch {epoch}] '
              f'L_CE_agree={mean_ce_agree:.4f}  L_CE_disagree={mean_ce_disagree:.4f}  L_balance={mean_balance:.4f}')
        print(f'  mean_cons_t={mean_cons_t:.4f}  mean_cons_c={mean_cons_c:.4f}  '
              f'mean_bdry_t={mean_bdry_t:.4f}  mean_bdry_c={mean_bdry_c:.4f}')
        print(f'  case1 (clean agreement): {total_case1:,}  '
              f'case2 (informative disagreement): {total_case2:,}  '
              f'case3 (ignored): {total_case3:,}  '
              f'skipped (too small): {total_skipped_small:,}')
        print(f'  case2 pixels this epoch: {total_case2_px:,}')
        print(f'  L_CE_agree unweighted={mean_ce_agree_unw:.4f}  L_partial={mean_partial:.4f}  '
              f'case1 text-rejected: {total_case1_partial:,} regions / {total_case1_partial_px:,} px  '
              f'case1 text-relabelled: {total_case1_relabel:,} regions / {total_case1_relabel_px:,} px  '
              f'blind-class: {total_blind:,} regions / {total_blind_px:,} px')
        if relabel_pairs:
            print('  relabelled px by pair: ' + '  '.join(f'{k}={n:,}' for k, n in relabel_pairs.items()))
        print('  class weights: ' + '  '.join(
            f'{n}={w:.2f}' for n, w, f in zip(CITYSCAPES_CLASSES, cb_w.tolist(), class_freq.tolist()) if f > 0))
        print(f'  proxy disagreement rate (target vs VLM, {n_eval} held-out train images): '
              f'{disagree_rate_eval:.4f}  (best={best_disagree_rate:.4f}, '
              f'no improvement for {epochs_no_improve}/{args.early_stop_patience} epochs)')

        checkpoint = {
            'epoch': epoch,
            'target_model': target_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'thresholds': thresholds,
            'proxy_disagree_rate': disagree_rate_eval,
            'best_disagree_rate': best_disagree_rate,
            'epochs_no_improve': epochs_no_improve,
            'class_freq': class_freq.cpu(),
            'text_z': (None if text_z is None else
                       {k: v.cpu() if torch.is_tensor(v) else v for k, v in text_z.items()}),
            'blind_classes': sorted(blind_classes),
            'args': vars(args),
        }
        if epoch == 1 or epoch % args.ckpt_interval == 0:
            ckpt_path = os.path.join(CKPT_DIR, f'dsp_epoch{epoch}.pth')
            torch.save(checkpoint, ckpt_path)
            print(f'  Saved -> {ckpt_path}')
        else:
            print(f'  Skipped periodic checkpoint this epoch (--ckpt-interval={args.ckpt_interval})')

        if is_best:
            best_ckpt_path = os.path.join(CKPT_DIR, 'best.pth')
            torch.save(checkpoint, best_ckpt_path)
            print(f'  New best proxy disagreement rate ({disagree_rate_eval:.4f}) -> saved {best_ckpt_path}')

        save_training_viz(target_model, val_sample, device, TRAIN_VIZ_DIR, epoch)

        writer.add_scalar('train_epoch/L_CE_agree', mean_ce_agree, epoch)
        writer.add_scalar('train_epoch/L_CE_disagree', mean_ce_disagree, epoch)
        writer.add_scalar('train_epoch/L_balance', mean_balance, epoch)
        writer.add_scalar('train_epoch/mean_cons_t', mean_cons_t, epoch)
        writer.add_scalar('train_epoch/mean_cons_c', mean_cons_c, epoch)
        writer.add_scalar('train_epoch/mean_bdry_t', mean_bdry_t, epoch)
        writer.add_scalar('train_epoch/mean_bdry_c', mean_bdry_c, epoch)
        writer.add_scalar('train_epoch/total_case1_regions', total_case1, epoch)
        writer.add_scalar('train_epoch/total_case2_regions', total_case2, epoch)
        writer.add_scalar('train_epoch/total_case3_regions', total_case3, epoch)
        writer.add_scalar('train_epoch/total_skipped_small_regions', total_skipped_small, epoch)
        writer.add_scalar('train_epoch/total_case2_px', total_case2_px, epoch)
        writer.add_scalar('train_epoch/proxy_disagree_rate', disagree_rate_eval, epoch)
        writer.flush()

        with open(TRAIN_LOG_PATH, 'a') as f:
            f.write(json.dumps({
                'epoch': epoch,
                'mean_ce_agree': mean_ce_agree, 'mean_ce_disagree': mean_ce_disagree,
                'mean_balance': mean_balance,
                'mean_cons_t': mean_cons_t, 'mean_cons_c': mean_cons_c,
                'mean_bdry_t': mean_bdry_t, 'mean_bdry_c': mean_bdry_c,
                'total_case1_regions': total_case1,
                'total_case2_regions': total_case2,
                'total_case3_regions': total_case3,
                'total_skipped_small_regions': total_skipped_small,
                'total_case2_px': total_case2_px,
                'proxy_disagree_rate': disagree_rate_eval,
                'mean_ce_agree_unweighted': mean_ce_agree_unw, 'mean_partial': mean_partial,
                'total_case1_partial_regions': total_case1_partial,
                'total_case1_partial_px': total_case1_partial_px,
                'total_case1_relabel_regions': total_case1_relabel,
                'total_case1_relabel_px': total_case1_relabel_px,
                'relabel_px_by_pair': relabel_pairs,
                'total_blind_regions': total_blind, 'total_blind_px': total_blind_px,
                'class_share': dict(zip(CITYSCAPES_CLASSES, class_freq.tolist())),
                'class_weights': dict(zip(CITYSCAPES_CLASSES, cb_w.tolist())),
            }) + '\n')

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f'\nProxy disagreement rate has not reached a new low for '
                  f'{args.early_stop_patience} consecutive epochs (best={best_disagree_rate:.4f}) '
                  f'— stopping early at epoch {epoch}.')
            break

    writer.close()
    print('\nRegion-level DSP (3-case) training complete.')


if __name__ == '__main__':
    main()
