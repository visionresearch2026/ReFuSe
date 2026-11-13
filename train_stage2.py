"""
ReFuSe Stage 2: refinement with a frozen Stage 1 reference and dual prototype banks.

One script for every setting; the experiment config (configs/*.yaml) picks
the source domain (gta5 | synthia), the backbone (deeplabv2 | mitb5) and the
VLM (corrclip | proxyclip). No target label is read anywhere.

The model is initialised from the Stage 1 best.pth, and the frozen Stage 1
argmax is the reference label in every epoch (written once to --ref-dir).
For every SAM2 region R (>= --min-region-size valid pixels):
  Case 1  reference == VLM label (pixel-level agreement, --case1-pixel-agree)
          -> hard label, class-balanced CE (--cb-power), text z-score
             verification of rare labels (--text-verify, as in Stage 1),
             VLM-blind classes from the source pseudo-label (rule C).
          Case 1 pixels update the dual prototype bank (EMA): target-feature
          prototypes (DeepLabV2 layer4 / MiT-B5 fuse_layer) and VLM-feature
          prototypes.
  Case 2  disagreement with reliable sides -> the prototype banks score the
          two candidate labels; a clear winner gives a sharpened soft label
          (L_CE_disagree), otherwise the partial label -log(Pt[y_s]+Pt[y_c]).
  Case 3  otherwise -> no supervision.
  L_compress   pulls the region-mean target feature to its class prototype
  L_consist    keeps region predictions consistent with the prototype evidence
  L = L_CE_agree + lambda_dis L_CE_disagree + lambda_comp L_compress
      + lambda_cons L_consist + lambda_partial L_partial + mu L_balance
Optional components (off in the main results of the paper):
  --proto-contrast on   prototype contrastive loss on target features (--lambda-pc)
  --proto-denoise on    relabel unsupervised pixels with frozen Stage 1 x VLM
                        prototype evidence (--lambda-relabel)

Model selection (no target labels): 200 training images are held out; after
every epoch the model is scored against the consensus pseudo-label (source
pseudo-label where it equals the VLM label). The highest consensus mIoU is
best.pth; training stops when the held-out CE has not decreased for 5
evaluations.

Determinism: with --deterministic (default) the non-deterministic CUDA ops
are replaced (logit upsample as matmuls, CE as log_softmax + gather, region
pooling with torch.unique). The MiT-B5 head still has non-deterministic
resizes in torch 1.10, so MiT-B5 reruns are close but not bit-identical.

Usage:
    python train_stage2.py --config configs/gta5_deeplabv2_corrclip.yaml --gpu 0
    python train_stage2.py --config ... --proto-contrast on --proto-denoise on
    python train_stage2.py --config ... --lambda-cons 0 --run-name stage2_noConsist
    python train_stage2.py --config ... --resume <work_dir>/<run>/dsp_stage_epoch05.pth
Defaults: --stage-c-ckpt <work_dir>/stage1/best.pth,
          --ref-dir <work_dir>/stage1_reference_labels/train,
          outputs <work_dir>/stage2_contrast<on|off>_denoise<on|off>/.
"""

import os
import sys
import csv
import json
import math
import random
import shutil
import argparse
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refuse.common import (NUM_CLASSES, IGNORE_LABEL, TARGET_H, TARGET_W,  # noqa: E402
                           CITYSCAPES_CLASSES, colorize)
from refuse.backbones import freeze_bn  # noqa: E402
from refuse.config import load_config, make_backbone, default_text_z_classes  # noqa: E402
from refuse.vlm import VLM  # noqa: E402
from refuse.dual_prototype_bank import DualPrototypeBank  # noqa: E402

# Set once in main() from the experiment config.
BB = None                  # refuse.backbones.Backbone
VLM_T = None               # refuse.vlm.VLM (the VLM teacher)
ABSENT_CLASSES = []        # classes the source domain does not have (SYNTHIA: terrain, truck, train)
ABSENT_T = None

PS_SUFFIX    = '_pseudo_label.png'
IMG_SUFFIX   = '_leftImg8bit.png'
MASK_SUFFIX  = '_leftImg8bit.npz'
EPS = 1e-8

# Fixed Case-2 thresholds / temperatures (saved alongside the calibrated ones).
THETA_FEAT   = 0.3
THETA_COMPAT = 0.1
TAU_HIGH = 0.3
TAU_LOW  = 0.7
# Recorded in thresholds.json: caches calibrated before cons_c/bdry_c moved to
# the raw Pc argmax have a different q_c distribution and are recalibrated.
Q_C_SOURCE = 'raw_pc_argmax'
# Recorded in thresholds.json: the reference is the frozen Stage 1 argmax.
REFERENCE_SOURCE = 'frozen_stage_c'
# Log files a fresh run starts from scratch and a resumed run truncates.
LOG_FILES = ('train_log.csv', 'epoch_summary.jsonl')

CSV_FIELDS = ['epoch', 'image_idx', 'city', 'stem',
              'n_case1', 'n_case2_high', 'n_case2_low', 'n_case3',
              'px_case1', 'px_case2', 'px_case3', 'ignored_frac',
              'mean_q_t', 'mean_q_c', 'mean_s_s', 'mean_s_c', 'mean_delta', 'mean_tilde_alpha',
              'L_CE_agree', 'L_CE_agree_unweighted', 'cb_weight_max',
              'L_CE_disagree', 'L_compress', 'L_consist', 'L_balance', 'L_total',
              'n_case2_noproto', 'n_case1_partial', 'px_case1_partial', 'L_partial',
              'L_proto_contrast', 'pc_cells', 'pc_acc',
              'L_CE_relabel', 'px_relabel', 'px_relabel_changed', 'n_case1_dropped', 'px_case1_dropped',
              'n_blind', 'px_blind', 'n_case1_textrelabel', 'px_case1_textrelabel']
# Per-image TensorBoard scalars (train_iter/<name>), a subset of CSV_FIELDS.
TB_ITER_FIELDS = ['L_total', 'L_CE_agree', 'L_CE_agree_unweighted', 'cb_weight_max', 'L_CE_disagree', 'L_compress', 'L_consist', 'L_balance',
                  'L_partial', 'n_case1_partial', 'L_proto_contrast', 'pc_cells', 'pc_acc',
                  'L_CE_relabel', 'px_relabel', 'px_relabel_changed', 'n_case1_dropped', 'px_case1_dropped',
                  'n_blind', 'px_blind', 'n_case1_textrelabel', 'px_case1_textrelabel', 'ignored_frac', 'n_case1', 'n_case2_high', 'n_case2_low', 'n_case2_noproto', 'n_case3',
                  'mean_q_t', 'mean_q_c', 'mean_s_s', 'mean_s_c', 'mean_delta', 'mean_tilde_alpha']
VAL_REFERENCES = ('consensus', 'pc', 'ps')
def setup_reproducibility(seed, deterministic):
    """Seed every RNG and, if deterministic, force deterministic GPU kernels.
    CUBLAS_WORKSPACE_CONFIG must be set before cuBLAS is first used."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                 # also seeds every CUDA device
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # torch < 1.11 has no warn_only; strict mode would raise on ops with no
            # deterministic kernel (e.g. bilinear upsample backward), so rely on the
            # cuDNN flags above only.
            pass


def stratified_holdout_split(samples, num_holdout, seed):
    """Hold out num_holdout samples, proportionally per city (largest-remainder
    rounding to hit the exact total). Returns (remaining, holdout). Same split
    as Stage 1 (same --seed and number of images)."""
    by_city = defaultdict(list)
    for s in samples:
        by_city[s['city']].append(s)
    total = len(samples)
    if not 0 < num_holdout < total:
        raise ValueError(f'num_holdout ({num_holdout}) must be in (0, {total})')

    quotas = {city: num_holdout * len(items) / total for city, items in by_city.items()}
    counts = {city: int(q) for city, q in quotas.items()}
    by_remainder = sorted(quotas.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)
    for i in range(num_holdout - sum(counts.values())):
        counts[by_remainder[i % len(by_remainder)][0]] += 1

    rng = np.random.default_rng(seed)
    remaining, holdout = [], []
    for city in sorted(by_city):
        items = by_city[city][:]
        rng.shuffle(items)
        holdout.extend(items[:counts[city]])
        remaining.extend(items[counts[city]:])
    return remaining, holdout


def fixed_viz_sample(holdout_entries, n):
    """Evenly spaced (city, filename) pairs from the held-out val images, for
    the per-epoch qualitative panels."""
    if not holdout_entries or n <= 0:
        return []
    idx = np.linspace(0, len(holdout_entries) - 1, num=min(n, len(holdout_entries)), dtype=int)
    return [(holdout_entries[i]['city'], os.path.basename(holdout_entries[i]['img_path'])) for i in idx]


def save_training_viz(model, val_sample, img_dir, device, out_dir, epoch):
    """Diagnostic only: image | colourised prediction panels for val_sample
    (held-out train images), saved to out_dir/epochNN/. Never affects
    training, checkpointing or best.pth selection. Returns [(name, BGR panel)]."""
    if not val_sample:
        return []
    epoch_dir = os.path.join(out_dir, f'epoch{epoch:02d}')
    os.makedirs(epoch_dir, exist_ok=True)
    panels = []
    model.eval()
    with torch.no_grad():
        for city, fname in val_sample:
            img = cv2.imread(os.path.join(img_dir, city, fname))
            if img is None:
                continue
            pred_colored = colorize(predict(model, img, device))
            h, w = img.shape[:2]
            panel = np.hstack([
                cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA),
                cv2.resize(pred_colored, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            ])
            name = f'{city}_{fname[:-len(IMG_SUFFIX)]}'
            cv2.imwrite(os.path.join(epoch_dir, f'{name}.png'), panel)
            panels.append((name, panel))
    model.train()
    freeze_bn(model)          # .eval() above also flipped BN; re-freeze it
    return panels


def log_viz_to_tensorboard(writer, panels, epoch):
    for name, panel in panels:
        writer.add_image(f'val_viz/{name}', panel[:, :, ::-1].copy(), epoch, dataformats='HWC')


def preprocess(img_bgr):
    return BB.preprocess(img_bgr)


def predict(model, img_bgr, device):
    return BB.predict(model, img_bgr, device)


def model_logits(model, x, out_hw):
    """forward(), with a deterministic (matmul) logit upsample. Training path
    only; eval/viz keep model.forward (forward is deterministic)."""
    return BB.model_logits(model, x, out_hw)


def load_model_weights(path):
    return BB.load_weights(path)


def collect_entries(args):
    entries = []
    for city in sorted(os.listdir(args.img_dir)):
        img_city = os.path.join(args.img_dir, city)
        if not os.path.isdir(img_city):
            continue
        for fname in sorted(os.listdir(img_city)):
            if not fname.endswith(IMG_SUFFIX):
                continue
            stem = fname[:-len(IMG_SUFFIX)]
            e = {
                'stem': stem, 'city': city,
                'img_path': os.path.join(img_city, fname),
                'ps_path': os.path.join(args.ps_dir, city, stem + PS_SUFFIX),      # rule (C) + held-out val
                'ref_path': os.path.join(args.ref_dir, city, stem + PS_SUFFIX),    # frozen Stage 1
                'mask_path': os.path.join(args.sam2_dir, city, stem + MASK_SUFFIX),
                **VLM_T.required_paths(city, stem),
            }
            required = ['ps_path', 'mask_path'] + list(VLM_T.required_paths(city, stem))
            if all(os.path.exists(e[k]) for k in required):
                entries.append(e)
    return entries


def entropy(p, dim=-1):
    return -(p * torch.log(p + EPS)).sum(dim=dim)


def temperature_soften(p, T, dim=-1):
    """p ** (1/T), renormalised -- softens the target model's confidence
    before q_t's entropy term (as in Stage 1)."""
    powered = p.clamp_min(EPS) ** (1.0 / T)
    return powered / powered.sum(dim=dim, keepdim=True)


_lowres_map_cache = {}


def lowres_index_map(h, w, device, align_corners=True):
    """(TARGET_H*TARGET_W,) flat index of the nearest (h, w) feature cell for
    every full-resolution pixel. align_corners=True geometry for the VLM
    feature map and the DeepLabV2 features, False (half-pixel, the geometry
    of the MiT-B5 logit resizes) for the MiT-B5 features."""
    key = (h, w, str(device), align_corners)
    if key not in _lowres_map_cache:
        if align_corners:
            ys = torch.round(torch.arange(TARGET_H, device=device, dtype=torch.float32)
                             * (h - 1) / (TARGET_H - 1)).long()
            xs = torch.round(torch.arange(TARGET_W, device=device, dtype=torch.float32)
                             * (w - 1) / (TARGET_W - 1)).long()
        else:
            ys = ((torch.arange(TARGET_H, device=device, dtype=torch.float32) + 0.5)
                  * h / TARGET_H).floor().long().clamp(max=h - 1)
            xs = ((torch.arange(TARGET_W, device=device, dtype=torch.float32) + 0.5)
                  * w / TARGET_W).floor().long().clamp(max=w - 1)
        _lowres_map_cache[key] = (ys[:, None] * w + xs[None, :]).reshape(-1)
    return _lowres_map_cache[key]


def region_feature(feat_flat, lowres_map, region_idx):
    """Mean feature of a full-resolution region, pooled on the low-res map.
    feat_flat: (C, h*w). Differentiable w.r.t. feat_flat. Cell counts come
    from torch.unique (sort-based, deterministic) rather than CUDA bincount."""
    cells, counts = torch.unique(lowres_map[region_idx], return_counts=True)
    weights = counts.to(feat_flat.dtype)
    return feat_flat[:, cells] @ weights / weights.sum()


def ensure_reference_labels(entries, ckpt_path, ref_dir, device):
    """Write the frozen Stage 1 model's argmax for every entry to
    entry['ref_path'] (uint8 PNG, same format as Ps), skipping files that
    already exist. reference_meta.json in ref_dir records the checkpoint;
    a mismatch raises instead of silently mixing labels of two models."""
    ckpt_path = os.path.abspath(ckpt_path)
    meta_path = os.path.join(ref_dir, 'reference_meta.json')
    os.makedirs(ref_dir, exist_ok=True)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta_ckpt = json.load(f)['stage_c_ckpt']
        if meta_ckpt != ckpt_path:
            raise ValueError(f'{ref_dir} holds reference labels of {meta_ckpt}, not {ckpt_path}; '
                             'pass another --ref-dir or delete that one.')
    else:
        with open(meta_path, 'w') as f:
            json.dump({'stage_c_ckpt': ckpt_path}, f, indent=2)

    missing = [e for e in entries if not os.path.exists(e['ref_path'])]
    print(f'Reference labels (frozen Stage 1): {len(entries) - len(missing)}/{len(entries)} '
          f'already in {ref_dir}, generating {len(missing)}')
    if not missing:
        return
    ref_model = BB.build(device, ckpt_path).eval()
    with torch.no_grad():
        for e in tqdm(missing, desc='reference labels'):
            img_bgr = cv2.imread(e['img_path'])
            if img_bgr is None:
                continue
            logits = ref_model(preprocess(img_bgr).to(device), out_size=(TARGET_H, TARGET_W))
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            os.makedirs(os.path.dirname(e['ref_path']), exist_ok=True)
            # Per-process temp name: two runs sharing --ref-dir may write the same
            # label at once (identical content); each renames its own file.
            tmp_path = e['ref_path'][:-len('.png')] + f'.tmp{os.getpid()}.png'
            if not cv2.imwrite(tmp_path, pred):
                raise IOError(f'could not write {tmp_path}')
            os.replace(tmp_path, e['ref_path'])       # no half-written label after a crash
    del ref_model
    torch.cuda.empty_cache()


def case1_pixels(maps, region_idx, y_s, agreement_strength, thresholds, args):
    """Case 1 test. Returns the region pixels that get hard label y_s, or
    None if the region is not Case 1 (the caller has checked y_s == y_c).
    With --case1-pixel-agree these are only the pixels where the reference
    and the SAM2-cleaned Pc_label both equal y_s; fewer than
    --min-region-size of them -> None."""
    if agreement_strength <= thresholds['theta_agree']:
        return None
    if not args.case1_pixel_agree:
        return region_idx
    keep = ((maps['reference_flat'][region_idx] == y_s)
            & (maps['Pc_label_flat'][region_idx] == y_s))
    pix = region_idx[keep]
    return pix if pix.numel() >= args.min_region_size else None


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
    text embedding space (cosine, all 19 classes). Everything comes from the
    current CLIP model and data, so a new model (NACLIP / SCLIP / ProxyCLIP)
    gets its own pairs. Returns ({y: [confusers]}, {y: share})."""
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


def text_verify(fc_region, y, text_embed, text_z):
    """Case 1 verification. Returns the confusers c of y (text_z['pairs'][y])
    whose z-scored text similarity beats y's by more than text_z['margin']
    (z_c - z_y > margin); empty = accept the hard label y.
    fc_region: pooled L2-normalised VLM feature (512,);
    text_z: {'mu', 'sd'} (19,) tensors from compute_text_z_stats, plus
    'pairs' and 'margin', fixed after the warm start."""
    if text_z is None:
        return []
    confusers = text_z['pairs'].get(y)
    if not confusers:
        return []
    z = (text_embed @ F.normalize(fc_region, dim=0) - text_z['mu']) / text_z['sd']
    return [c for c in confusers if z[c] - z[y] > text_z['margin']]


def text_relabel_target(fc_region, y, rejecting, text_embed, text_z):
    """Hard-relabel target of a text-rejected Case 1 region: the rejecting
    confuser c with the largest z(c) - z(y), if that gap exceeds
    text_z['relabel_margin'] and c is not in text_z['no_relabel'] (the
    VLM-blind classes, whose text similarity means nothing); else None."""
    if not rejecting:
        return None
    z = (text_embed @ F.normalize(fc_region, dim=0) - text_z['mu']) / text_z['sd']
    gap, c = max(((z[c] - z[y]).item(), c) for c in rejecting)
    if gap > text_z.get('relabel_margin', float('inf')) and c not in text_z.get('no_relabel', ()):
        return c
    return None


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

def compute_image_maps(model, feat_cache, entry, text_embed, device, ref_model=None):
    """Forward pass + per-image maps shared by every region. Returns None if
    any input file is unreadable/malformed. With ref_model (IDEA 2) the frozen
    Stage 1 probabilities are added as 'Pref_probs_flat'."""
    img_bgr = cv2.imread(entry['img_path'])
    if img_bgr is None:
        return None
    instance_mask_np = np.load(entry['mask_path'])['instance_mask']
    if instance_mask_np.shape != (TARGET_H, TARGET_W):
        return None
    pc = VLM_T.maps(entry, device)       # frozen VLM: label, probs, L2-normalised features, raw argmax
    if pc is None:
        return None
    Pc_label = pc['label']
    instance_mask = torch.from_numpy(instance_mask_np.astype(np.int64)).to(device)

    x = preprocess(img_bgr).to(device)
    logits = model_logits(model, x, (TARGET_H, TARGET_W))                                    # (1,19,H,W)
    Pt_probs = F.softmax(logits, dim=1)
    ft = feat_cache['ft']                                                                     # (1,C,h',w')
    Pref_flat = None
    if ref_model is not None:            # the hook is on the trained model only; ft is not overwritten
        with torch.no_grad():
            Pref_flat = F.softmax(ref_model(x, out_size=(TARGET_H, TARGET_W)), dim=1)[0].reshape(NUM_CLASSES, -1)

    # Reference = frozen Stage 1 argmax, every epoch.
    ref_np = cv2.imread(entry['ref_path'], cv2.IMREAD_GRAYSCALE)
    if ref_np is None or ref_np.shape != (TARGET_H, TARGET_W):
        return None
    reference_label = torch.from_numpy(ref_np.astype(np.int64)).to(device)
    # Frozen source pseudo-label Ps: the anchor of the VLM-blind class rule (C).
    ps_np = cv2.imread(entry['ps_path'], cv2.IMREAD_GRAYSCALE)
    if ps_np is None or ps_np.shape != (TARGET_H, TARGET_W):
        ps_np = np.full((TARGET_H, TARGET_W), IGNORE_LABEL, dtype=np.uint8)
    ps_label = torch.from_numpy(ps_np.astype(np.int64)).to(device)

    fc_norm, Pc_probs_map = pc['feat'], pc['probs']                                            # (512,h,w), (19,H,W)
    # Raw (pre-SAM2-voting) VLM prediction, for the consensus/boundary terms of q_c.
    Pc_raw_label = pc['raw_label']

    valid = Pc_label < NUM_CLASSES
    return {
        'logits': logits,
        'Pt_probs_flat': Pt_probs[0].reshape(NUM_CLASSES, -1),                  # keeps grad
        'ft_flat': ft[0].reshape(ft.shape[1], -1),                               # (C, h'*w'), keeps grad
        'ft_hw': (ft.shape[2], ft.shape[3]),
        'ft_lowres_map': lowres_index_map(ft.shape[2], ft.shape[3], device, BB.feat_align_corners),
        'fc_flat': fc_norm.reshape(fc_norm.shape[0], -1),                        # (512, h*w), detached
        'fc_lowres_map': lowres_index_map(fc_norm.shape[1], fc_norm.shape[2], device),
        'reference_flat': reference_label.reshape(-1),
        'ps_flat': ps_label.reshape(-1),                                         # source Ps: rule (C) only
        'Pref_probs_flat': Pref_flat,                                            # frozen Stage 1 probs (IDEA 2)
        'Pc_label_flat': Pc_label.reshape(-1),                                   # SAM2-cleaned: Case 1 only
        'Pc_raw_label_flat': Pc_raw_label.reshape(-1),                           # raw argmax: cons_c, bdry_c
        'Pc_probs_flat': Pc_probs_map.reshape(NUM_CLASSES, -1),
        'instance_mask_flat': instance_mask.reshape(-1),
        'valid_flat': valid.reshape(-1),
    }


def get_region_indices(instance_mask_flat, valid_flat, min_region_pixels):
    region_ids = torch.unique(instance_mask_flat[valid_flat])
    regions, n_skipped = [], 0
    for rid in region_ids.tolist():
        region_idx = torch.nonzero((instance_mask_flat == rid) & valid_flat, as_tuple=True)[0]
        if region_idx.numel() < min_region_pixels:
            n_skipped += 1
            continue
        regions.append(region_idx)
    return regions, n_skipped


def compute_boundary_agreement(region_idx, maps, y_s, y_c, ref_key='reference_flat'):
    """Share of the region's boundary pixels (4-neighbourhood) whose label
    matches that model's regional majority. Always on the full region.
    VLM's side uses its raw argmax, not the SAM2-cleaned Pc_label."""
    H, W = TARGET_H, TARGET_W
    region_mask = torch.zeros(H, W, dtype=torch.bool, device=region_idx.device)
    region_mask[region_idx // W, region_idx % W] = True

    up = torch.zeros_like(region_mask)
    down = torch.zeros_like(region_mask)
    left = torch.zeros_like(region_mask)
    right = torch.zeros_like(region_mask)
    up[1:, :] = region_mask[:-1, :]
    down[:-1, :] = region_mask[1:, :]
    left[:, 1:] = region_mask[:, :-1]
    right[:, :-1] = region_mask[:, 1:]
    boundary = region_mask & (~up | ~down | ~left | ~right)

    bdry_rows, bdry_cols = torch.where(boundary)
    if len(bdry_rows) == 0:
        return 1.0, 1.0
    bdry_flat = bdry_rows * W + bdry_cols
    bdry_t = (maps[ref_key][bdry_flat] == y_s).float().mean().item()
    bdry_c = (maps['Pc_raw_label_flat'][bdry_flat] == y_c).float().mean().item()
    return bdry_t, bdry_c


def region_stats(sample_idx, maps, region_idx, qt_temperature):
    """STEPS 2-5 (Stage 1). Stats on sample_idx (capped subsample for large
    regions); boundary agreement on the full region.

    y_c and agreement_strength (the Case 1 decision) use the SAM2-cleaned
    Pc_label; cons_c and bdry_c (VLM's reliability q_c) use its raw
    argmax, since the cleaned label is uniform within a region by
    construction and would inflate both towards 1."""
    reference_votes = maps['reference_flat'][sample_idx]
    Pc_votes = maps['Pc_label_flat'][sample_idx]
    Pc_raw_votes = maps['Pc_raw_label_flat'][sample_idx]
    with torch.no_grad():
        Pt_mean = maps['Pt_probs_flat'][:, sample_idx].mean(dim=1)
    Pc_mean = maps['Pc_probs_flat'][:, sample_idx].mean(dim=1)

    y_s = reference_votes.mode().values.item()
    y_c = Pc_votes.mode().values.item()
    agreement_strength = (reference_votes == Pc_votes).float().mean().item()
    cons_t = (reference_votes == y_s).float().mean().item()
    cons_c = (Pc_raw_votes == y_c).float().mean().item()
    bdry_t, bdry_c = compute_boundary_agreement(region_idx, maps, y_s, y_c)

    q_t = (math.exp(-entropy(temperature_soften(Pt_mean, qt_temperature)).item()) + cons_t + bdry_t) / 3.0
    q_c = (math.exp(-entropy(Pc_mean).item()) + cons_c + bdry_c) / 3.0
    return {'y_s': y_s, 'y_c': y_c, 'agreement_strength': agreement_strength,
            'q_t': q_t, 'q_c': q_c, 'Pt_mean': Pt_mean, 'Pc_mean': Pc_mean}


# ── (C) VLM-blind classes (as in Stage 1) ─────────────────────────────────────
def blind_class_trust(sample_idx, region_idx, maps, rs, blind_classes, args):
    """Rule (C). Returns (k, q_src, pix) if the frozen source pseudo-label Ps
    votes a VLM-blind class k for this region, VLM votes something else,
    and q_src = (exp(-H_t) + cons_ps + bdry_ps) / 3 > --blind-q-thresh; pix =
    the region's Ps == k pixels (the ones that get the hard label k). None
    otherwise."""
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
    if pix.numel() < args.min_region_size:
        return None
    return k, q_src, pix


def blind_class_pixels(maps, region_idx, rng, args, device):
    """Rule (C) for the no-grad passes (warm start, relabel calibration):
    (k, pix) if the region is a trusted blind-class region, else None."""
    if not args.blind_class_ids:
        return None
    sample_idx = sample_region(region_idx, args.max_region_sample_pixels, rng, device)
    rs = region_stats(sample_idx, maps, region_idx, args.qt_temperature)
    trust = blind_class_trust(sample_idx, region_idx, maps, rs, args.blind_class_ids, args)
    return None if trust is None else (trust[0], trust[2])


def sample_region(region_idx, cap, rng, device):
    if region_idx.numel() > cap:
        sub = torch.from_numpy(rng.choice(region_idx.numel(), cap, replace=False)).to(device)
        return region_idx[sub]
    return region_idx


def calibrate_thresholds(model, feat_cache, calib_entries, text_embed, args, device):
    """Stage 1 calibration: percentiles of agreement_strength, max(q_t,q_c)
    and |q_t-q_c| over every region of the calibration images."""
    print(f'\nCalibrating thresholds on {len(calib_entries)} training images...')
    rng = np.random.default_rng(args.seed)
    agree_vals, max_q_vals, margin_vals = [], [], []
    with torch.no_grad():
        for entry in tqdm(calib_entries, desc='calibration'):
            maps = compute_image_maps(model, feat_cache, entry, text_embed, device)
            if maps is None:
                continue
            regions, _ = get_region_indices(maps['instance_mask_flat'], maps['valid_flat'],
                                            args.min_region_size)
            for region_idx in regions:
                sample_idx = sample_region(region_idx, args.max_region_sample_pixels, rng, device)
                rs = region_stats(sample_idx, maps, region_idx, args.qt_temperature)
                agree_vals.append(rs['agreement_strength'])
                max_q_vals.append(max(rs['q_t'], rs['q_c']))
                margin_vals.append(abs(rs['q_t'] - rs['q_c']))

    thresholds = {
        'theta_agree': float(np.percentile(agree_vals, 60)),
        'theta_rel': float(np.percentile(max_q_vals, 30)),
        'theta_margin': float(np.percentile(margin_vals, 25)),
        'theta_feat': THETA_FEAT,
        'theta_compat': THETA_COMPAT,
        'n_calibration_images': len(calib_entries),
        'n_calibration_regions': len(agree_vals),
    }
    print(f'Calibration done on {len(agree_vals)} regions: '
          + '  '.join(f'{k}={v:.4f}' for k, v in thresholds.items() if k.startswith('theta')))
    return thresholds


def clip_evidence(fc, ev):
    """IDEA 2 -- pseudo-label evidence from the VLM visual prototypes, in
    the VLM feature space only: w = softmax_k(cos(fc, C_k) / temperature).
    fc: pooled VLM feature (512,) or (N, 512); returns (19,) or (N, 19).
    A class without prototype gets the mean cosine of the others (neutral)."""
    cos = F.normalize(fc, dim=-1) @ ev['protos'].T
    neutral = cos[..., ev['ready']].mean(dim=-1, keepdim=True)
    cos = torch.where(ev['ready'], cos, neutral.expand_as(cos))
    return F.softmax(cos / ev['temperature'], dim=-1)


def case1_dropped(fc_region, y, ev):
    """IDEA 2(b): a Case 1 region labelled y loses its hard label when the
    VLM prototypes give y less evidence than the calibrated per-label
    threshold (the lowest --denoise-c1-drop pixel share of that label)."""
    return bool(clip_evidence(fc_region, ev)[y] < ev['c1_threshold'][y])


def build_clip_evidence(case1_regions, args, device):
    """Frozen VLM prototypes for IDEA 2 and the Case 1 drop thresholds.
    VLM's features and the Stage 1 reference are both frozen, so the
    Case 1 regions and their VLM features never change: the prototypes
    are plain per-class means (one vote per region) of the accepted warm-start
    Case 1 regions, computed once, instead of an EMA.
    case1_regions: [(y, n_pix, ft_region, fc_region on device)]."""
    labels = torch.tensor([r[0] for r in case1_regions], device=device)
    n_pix = torch.tensor([r[1] for r in case1_regions], dtype=torch.float64, device=device)
    feats = F.normalize(torch.stack([r[3] for r in case1_regions]).float(), dim=1)
    protos = torch.zeros(NUM_CLASSES, feats.shape[1], device=device)
    ready = torch.zeros(NUM_CLASSES, dtype=torch.bool, device=device)
    for k in range(NUM_CLASSES):
        m = labels == k
        if int(m.sum()) >= args.denoise_min_regions:
            protos[k] = F.normalize(feats[m].mean(dim=0), dim=0)
            ready[k] = True
    if int(ready.sum()) < 2:
        raise RuntimeError('--proto-denoise on: fewer than 2 classes have --denoise-min-regions '
                           'warm-start Case 1 regions (use more --warmstart-images).')
    ev = {'protos': protos, 'ready': ready, 'temperature': args.denoise_temperature,
          'c1_threshold': torch.full((NUM_CLASSES,), float('-inf'), device=device),
          'relabel_threshold': float('inf')}
    if args.denoise_c1_drop > 0:
        score = clip_evidence(feats, ev).gather(1, labels[:, None])[:, 0]
        for k in torch.nonzero(ready, as_tuple=True)[0].tolist():
            m = labels == k
            s_k, order = torch.sort(score[m])
            cum = torch.cumsum(n_pix[m][order], dim=0)
            # First region that is kept: dropping everything below it removes at
            # most --denoise-c1-drop of this label's Case 1 pixels.
            first_kept = int(torch.searchsorted(cum, args.denoise_c1_drop * cum[-1], right=True))
            ev['c1_threshold'][k] = s_k[min(first_kept, len(s_k) - 1)]
    print('VLM evidence prototypes: ' + ', '.join(
        f'{CITYSCAPES_CLASSES[k]}' for k in range(NUM_CLASSES) if ready[k])
          + ' | none for: ' + (', '.join(CITYSCAPES_CLASSES[k] for k in range(NUM_CLASSES) if not ready[k]) or '-'))
    return ev


def unsupervised_units(maps, sup_mask_flat, ev):
    """IDEA 2(a): labels for the pixels nothing else supervises. Every SAM2
    region's unsupervised pixels are split by their frozen Stage 1 label into
    units; a unit gets
        label = argmax(mean P_stageC over the unit * w),  conf = that maximum
    with w = clip_evidence of the unit's pooled VLM feature (the
    prototype-rectified pseudo-label of ProDA, with the prototypes living in
    the independent VLM space). Returns [(pix, label, conf, stage_c_label)]."""
    units = []
    unsup = maps['valid_flat'] & ~sup_mask_flat
    inst, ref = maps['instance_mask_flat'], maps['reference_flat']
    for rid in torch.unique(inst[unsup]).tolist():
        region_idx = torch.nonzero((inst == rid) & unsup, as_tuple=True)[0]
        labels = ref[region_idx]
        for l in torch.unique(labels).tolist():
            pix = region_idx[labels == l]
            w = clip_evidence(region_feature(maps['fc_flat'], maps['fc_lowres_map'], pix), ev)
            p = maps['Pref_probs_flat'][:, pix].mean(dim=1) * w
            conf, label = (p / p.sum().clamp_min(EPS)).max(dim=0)
            units.append((pix, int(label), float(conf), l))
    return units


def calibrate_relabel_threshold(model, feat_cache, ref_model, calib_entries, text_embed,
                                thresholds, text_z, ev, args, device):
    """Confidence threshold of IDEA 2(a): the units above it hold
    --denoise-coverage of the unsupervised pixels of calib_entries (no GT).
    Unsupervised here = everything but the kept / partial-label Case 1
    pixels; Case 2 (~0.4% of the pixels) is not excluded."""
    print(f'Calibrating the relabel threshold on {len(calib_entries)} training images...')
    rng = np.random.default_rng(args.seed + 3)
    blind_rng = np.random.default_rng(args.seed + 5)       # separate stream: rule (C) off leaves rng untouched
    confs, n_px = [], []
    with torch.no_grad():
        for entry in tqdm(calib_entries, desc='relabel calibration'):
            maps = compute_image_maps(model, feat_cache, entry, text_embed, device, ref_model)
            if maps is None:
                continue
            regions, _ = get_region_indices(maps['instance_mask_flat'], maps['valid_flat'],
                                            args.min_region_size)
            sup_mask_flat = torch.zeros_like(maps['valid_flat'])
            for region_idx in regions:
                blind = blind_class_pixels(maps, region_idx, blind_rng, args, device)
                if blind is not None:
                    sup_mask_flat[blind[1]] = True
                    continue
                sample_idx = sample_region(region_idx, args.max_region_sample_pixels, rng, device)
                reference_votes = maps['reference_flat'][sample_idx]
                Pc_votes = maps['Pc_label_flat'][sample_idx]
                y_s = reference_votes.mode().values.item()
                if y_s != Pc_votes.mode().values.item():
                    continue
                agreement_strength = (reference_votes == Pc_votes).float().mean().item()
                pix = case1_pixels(maps, region_idx, y_s, agreement_strength, thresholds, args)
                if pix is None:
                    continue
                fc_region = region_feature(maps['fc_flat'], maps['fc_lowres_map'], pix)
                if not text_verify(fc_region, y_s, text_embed, text_z) and case1_dropped(fc_region, y_s, ev):
                    continue
                sup_mask_flat[pix] = True
            for pix, _, conf, _ in unsupervised_units(maps, sup_mask_flat, ev):
                confs.append(conf)
                n_px.append(pix.numel())
    order = np.argsort(confs)[::-1]
    cum = np.cumsum(np.asarray(n_px, dtype=np.float64)[order])
    last_kept = min(int(np.searchsorted(cum, args.denoise_coverage * cum[-1])), len(order) - 1)
    return float(np.asarray(confs)[order][last_kept])


def warmstart_prototypes(model, feat_cache, proto_bank, warm_entries, text_embed,
                         thresholds, args, device):
    """Pre-build both prototype banks before epoch 1: one no-grad pass over
    warm_entries with the frozen Stage 1 reference, feeding every Case 1
    region (case1_pixels) into the bank exactly as training does. Without this, Case 2
    regions are demoted (or scored with placeholder values) until enough
    Case 1 regions have been seen online. Also returns the (NUM_CLASSES,)
    share of Case 1 pixels per class, the initial class frequency for the
    class-balanced L_CE_agree, and the text z-score statistics (None without
    --text-verify).

    With --text-verify the Case 1 regions are collected first (features kept
    on the CPU), the z-score statistics are computed from them, and only the
    regions text_verify accepts update the banks and the class share -- the
    same rule training applies."""
    print(f'\nWarm-starting prototypes on {len(warm_entries)} training images...')
    rng = np.random.default_rng(args.seed + 1)
    blind_rng = np.random.default_rng(args.seed + 4)       # separate stream: rule (C) off leaves rng untouched
    blind_px = defaultdict(int)                            # rule (C) pixels per blind class
    regions_out = []          # (y_s, n_pix, ft_region, fc_region), CPU
    with torch.no_grad():
        for entry in tqdm(warm_entries, desc='warmstart'):
            maps = compute_image_maps(model, feat_cache, entry, text_embed, device)
            if maps is None:
                continue
            regions, _ = get_region_indices(maps['instance_mask_flat'], maps['valid_flat'],
                                            args.min_region_size)
            for region_idx in regions:
                blind = blind_class_pixels(maps, region_idx, blind_rng, args, device)
                if blind is not None:
                    blind_px[blind[0]] += blind[1].numel()
                    continue
                sample_idx = sample_region(region_idx, args.max_region_sample_pixels, rng, device)
                reference_votes = maps['reference_flat'][sample_idx]
                Pc_votes = maps['Pc_label_flat'][sample_idx]
                y_s = reference_votes.mode().values.item()
                if y_s != Pc_votes.mode().values.item():
                    continue
                agreement_strength = (reference_votes == Pc_votes).float().mean().item()
                pix = case1_pixels(maps, region_idx, y_s, agreement_strength, thresholds, args)
                if pix is None:
                    continue
                regions_out.append((
                    y_s, pix.numel(),
                    region_feature(maps['ft_flat'], maps['ft_lowres_map'], pix).cpu(),
                    region_feature(maps['fc_flat'], maps['fc_lowres_map'], pix).cpu()))

    text_z = None
    if args.text_verify and regions_out:
        labels = [r[0] for r in regions_out]
        z_classes = set(args.text_z_class_ids)
        text_z = compute_text_z_stats(torch.stack([r[3] for r in regions_out]),
                                      labels, z_classes, text_embed)
        print(f"Text z-score stats from {text_z['n_regions']} warm-start Case 1 regions")
        if args.text_verify_pairs == 'auto':
            pairs, share = auto_verify_pairs(text_embed.cpu(), labels, [r[1] for r in regions_out],
                                             z_classes, args.text_verify_topk,
                                             args.text_verify_max_share)
            print('Case 1 pixel share within --text-z-classes: ' + '  '.join(
                f'{CITYSCAPES_CLASSES[y]}={v:.4f}' for y, v in sorted(share.items())))
        else:
            pairs = parse_verify_pairs(args.text_verify_pairs)
        text_z['pairs'] = pairs
        text_z['margin'] = args.text_verify_margin
        text_z['relabel_margin'] = (args.text_relabel_margin if args.text_verify_action == 'relabel'
                                    else float('inf'))
        text_z['no_relabel'] = list(args.blind_class_ids)
        print(f"Text relabel margin {text_z['relabel_margin']}, never relabel to: "
              + (', '.join(CITYSCAPES_CLASSES[k] for k in text_z['no_relabel']) or 'none'))
        print(f"Text verification pairs ({args.text_verify_pairs}, margin {args.text_verify_margin}): "
              + ('  '.join(f'{CITYSCAPES_CLASSES[y]}->{"/".join(CITYSCAPES_CLASSES[c] for c in cs)}'
                           for y, cs in sorted(pairs.items())) or 'none'))

    case1_px = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
    n_checked = defaultdict(int)
    n_rejected = defaultdict(int)
    n_relabelled = defaultdict(int)
    accepted = []
    for y_s, n_pix, ft_region, fc_region in regions_out:
        fc_region = fc_region.to(device)
        if text_z is not None and y_s in text_z['pairs']:
            n_checked[y_s] += 1
            rejecting = text_verify(fc_region, y_s, text_embed, text_z)
            if rejecting:
                n_rejected[y_s] += 1
                target = text_relabel_target(fc_region, y_s, rejecting, text_embed, text_z)
                if target is not None:         # hard label of the confuser: counts in the class share
                    case1_px[target] += n_pix
                    n_relabelled[(y_s, target)] += 1
                continue
        accepted.append((y_s, n_pix, ft_region, fc_region))
    # IDEA 2: frozen VLM evidence prototypes from the accepted Case 1 regions;
    # the regions it drops update neither bank nor the class share (as in training).
    ev = build_clip_evidence(accepted, args, device) if args.proto_denoise == 'on' else None
    n_dropped = defaultdict(int)
    for y_s, n_pix, ft_region, fc_region in accepted:
        if ev is not None and case1_dropped(fc_region, y_s, ev):
            n_dropped[y_s] += 1
            continue
        case1_px[y_s] += n_pix
        proto_bank.update_target(ft_region.to(device), y_s)
        proto_bank.update_clip(fc_region, y_s)
    if ev is not None and args.denoise_c1_drop > 0:
        n_label = defaultdict(int)
        for r in accepted:
            n_label[r[0]] += 1
        print('Warm-start Case 1 drop (dropped/accepted regions): ' + '  '.join(
            f'{CITYSCAPES_CLASSES[y]}={n_dropped[y]}/{n_label[y]}' for y in sorted(n_label)))
    if text_z is not None:
        print('Warm-start text verification (rejected/checked Case 1 regions): ' + '  '.join(
            f'{CITYSCAPES_CLASSES[y]}={n_rejected[y]}/{n_checked[y]}' for y in sorted(text_z['pairs'])))
    for k, n_pix in blind_px.items():          # rule (C) hard labels count in the class share
        case1_px[k] += n_pix
    if args.blind_class_ids:
        print(f'Warm-start blind-class pixels (q_src > {args.blind_q_thresh}): ' + '  '.join(
            f'{CITYSCAPES_CLASSES[k]}={blind_px[k]:,}' for k in sorted(args.blind_class_ids)))
    if n_relabelled:
        print('Warm-start text relabel (regions): ' + '  '.join(
            f'{CITYSCAPES_CLASSES[y]}->{CITYSCAPES_CLASSES[c]}={n}' for (y, c), n in sorted(n_relabelled.items())))
    proto_bank.summary()
    return (case1_px / case1_px.sum().clamp_min(1)).float(), text_z, ev


def calibration_config(args, init_path):
    """Everything the calibrated thresholds depend on. A cached thresholds.json
    is reused only if this matches exactly, so thresholds always follow from
    the command line rather than from whatever an earlier run left behind."""
    return {
        'q_c_source': Q_C_SOURCE,
        'reference_source': REFERENCE_SOURCE,
        'stage_c_ckpt': os.path.abspath(args.stage_c_ckpt),
        'init_ckpt': os.path.abspath(init_path),
        'seed': args.seed,
        'calib_images': args.calib_images,
        'min_region_size': args.min_region_size,
        'qt_temperature': args.qt_temperature,
        'max_region_sample_pixels': args.max_region_sample_pixels,
        'debug': args.debug,
        'holdout_images': args.holdout_images,   # calibration samples from the train split only
        **{k: os.path.abspath(getattr(args, k)) if getattr(args, k) else None
           for k in ('img_dir', 'ref_dir', 'sam2_dir')},
        **VLM_T.calib_paths(),
    }


def prepare_logs(ckpt_dir, resume_epoch):
    """Fresh run (resume_epoch None): move an earlier run's logs aside to
    <name>.prev so this run's logs contain only this run. Resumed run: drop
    rows after resume_epoch, which a crashed run may have written and the
    resumed run is about to re-train."""
    for name in LOG_FILES:
        path = os.path.join(ckpt_dir, name)
        if not os.path.exists(path):
            continue
        if resume_epoch is None:
            os.replace(path, path + '.prev')
            print(f'Moved earlier run log {path} -> {path}.prev')
            continue
        with open(path) as f:
            lines = f.readlines()
        if name.endswith('.csv'):
            header, rows = lines[:1], lines[1:]
            kept = header + [ln for ln in rows if int(ln.split(',', 1)[0]) <= resume_epoch]
        else:
            kept = [ln for ln in lines if ln.strip() and json.loads(ln)['epoch'] <= resume_epoch]
        if len(kept) != len(lines):
            with open(path, 'w') as f:
                f.writelines(kept)
            print(f'Truncated {path} to epochs <= {resume_epoch} ({len(lines) - len(kept)} rows dropped)')


def open_tensorboard(ckpt_dir, start_epoch, iters_per_epoch):
    """Writers for tensorboard/iter (step = global image index) and
    tensorboard/epoch (step = epoch). Fresh run: an earlier run's
    tensorboard/ is moved to tensorboard.prev. Resumed run: purge_step hides
    the events a crashed/longer earlier run wrote after the resumed epoch."""
    tb_dir = os.path.join(ckpt_dir, 'tensorboard')
    if start_epoch == 1 and os.path.exists(tb_dir):
        shutil.rmtree(tb_dir + '.prev', ignore_errors=True)
        os.replace(tb_dir, tb_dir + '.prev')
        print(f'Moved earlier run TensorBoard logs {tb_dir} -> {tb_dir}.prev')
    iter_writer = SummaryWriter(os.path.join(tb_dir, 'iter'),
                                purge_step=(start_epoch - 1) * iters_per_epoch)
    epoch_writer = SummaryWriter(os.path.join(tb_dir, 'epoch'), purge_step=start_epoch)
    return iter_writer, epoch_writer


def confusion_scores(conf):
    """(mIoU %, aAcc %, per-class IoU %, class present) from a flat
    (NUM_CLASSES*NUM_CLASSES,) confusion count [reference * C + prediction].
    mIoU averages over classes present in the reference, as in
    SegmentationMetrics."""
    conf = conf.view(NUM_CLASSES, NUM_CLASSES).double().cpu()
    tp = conf.diag()
    ref_px, pred_px = conf.sum(dim=1), conf.sum(dim=0)
    iou = tp / (ref_px + pred_px - tp).clamp_min(1)
    present = ref_px > 0
    miou = float(iou[present].mean()) * 100 if present.any() else float('nan')
    aacc = float(tp.sum() / conf.sum().clamp_min(1)) * 100
    return miou, aacc, (iou * 100).tolist(), present.tolist()


def evaluate_holdout(model, holdout_entries, device, val_label):
    """Score the model on the held-out val images against pseudo-labels only
    (no ground truth): the SAM2-cleaned VLM label 'pc', the source
    pseudo-label 'ps', and their 'consensus' (pixels where they agree).
    Returns flat val_* metrics (mIoU/aAcc against every reference, CE loss
    against val_label, consensus coverage) plus per-class IoU against
    val_label."""
    conf = {ref: torch.zeros(NUM_CLASSES * NUM_CLASSES, dtype=torch.long, device=device)
            for ref in VAL_REFERENCES}
    loss_sum, loss_px = 0.0, 0
    model.eval()
    with torch.no_grad():
        for entry in tqdm(holdout_entries, desc='val (held-out)', leave=False):
            img_bgr = cv2.imread(entry['img_path'])
            pc = VLM_T.label(entry, device)
            Ps_np = cv2.imread(entry['ps_path'], cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or pc is None or Ps_np is None or Ps_np.shape != (TARGET_H, TARGET_W):
                continue
            pc = pc.reshape(-1)
            ps = torch.from_numpy(Ps_np.astype(np.int64)).to(device).reshape(-1)
            pc = torch.where((pc < NUM_CLASSES) & ~torch.isin(pc, ABSENT_T),
                             pc, torch.full_like(pc, IGNORE_LABEL))   # no source-absent classes
            ps = torch.where(ps < NUM_CLASSES, ps, torch.full_like(ps, IGNORE_LABEL))
            refs = {'pc': pc, 'ps': ps,
                    'consensus': torch.where(pc == ps, pc, torch.full_like(pc, IGNORE_LABEL))}

            logits = model(preprocess(img_bgr).to(device), out_size=(TARGET_H, TARGET_W))
            logits = logits[0].reshape(NUM_CLASSES, -1)
            pred = logits.argmax(dim=0)
            for ref, label in refs.items():
                keep = label != IGNORE_LABEL
                conf[ref] += torch.bincount(label[keep] * NUM_CLASSES + pred[keep],
                                            minlength=NUM_CLASSES * NUM_CLASSES)
            # CE written out (log_softmax + gather) as in training: CUDA
            # nll_loss's reduction is not deterministic.
            pix = torch.nonzero(refs[val_label] != IGNORE_LABEL, as_tuple=True)[0]
            if pix.numel() > 0:
                log_probs = F.log_softmax(logits[:, pix], dim=0)
                loss_sum += float(-log_probs.gather(0, refs[val_label][pix][None]).sum())
                loss_px += pix.numel()
    model.train()
    freeze_bn(model)          # .eval() above also flipped BN; re-freeze it

    results = {'val_loss': loss_sum / max(1, loss_px),
               'val_consensus_coverage': float(conf['consensus'].sum()) / max(1, float(conf['pc'].sum()))}
    for ref in VAL_REFERENCES:
        miou, aacc, per_class, present = confusion_scores(conf[ref])
        results[f'val_miou_{ref}'] = miou
        results[f'val_aacc_{ref}'] = aacc
        if ref == val_label:
            results['per_class_iou'], results['class_present'] = per_class, present
    return results


def log_val_to_tensorboard(writer, results, epoch):
    for k, v in results.items():
        if k.startswith('val_'):
            writer.add_scalar(f'val/{k[4:]}', v, epoch)
    for name, iou, present in zip(CITYSCAPES_CLASSES, results['per_class_iou'], results['class_present']):
        if present:
            writer.add_scalar(f'val_class_iou/{name}', iou, epoch)


def save_run_config(ckpt_dir, args, start_epoch):
    """Command line, resolved arguments and software/hardware versions."""
    config = {
        'command': ' '.join([sys.executable] + sys.argv),
        'args': vars(args),
        'start_epoch': start_epoch,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'numpy': np.__version__, 'opencv': cv2.__version__,
        'gpu': torch.cuda.get_device_name(args.gpu), 'hostname': os.uname().nodename,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
    }
    name = 'run_config.json' if start_epoch == 1 else f'run_config_resume_epoch{start_epoch:02d}.json'
    with open(os.path.join(ckpt_dir, name), 'w') as f:
        json.dump(config, f, indent=2)


def nanmean(values):
    return float(np.mean(values)) if values else float('nan')


def masked_ce(logits_flat, target_flat, class_weights=None):
    """= F.cross_entropy(..., weight=class_weights, ignore_index=IGNORE_LABEL),
    written out because CUDA nll_loss2d's reduction is not deterministic.
    With class_weights the loss is sum(w[y] * nll) / sum(w[y]). None if no
    target."""
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
    for classes seen in Case 1 (f_c > 0); 1 for the others. power 0 -> all 1."""
    w = torch.ones_like(class_freq)
    seen = class_freq > 0
    if power > 0 and seen.any():
        w[seen] = (class_freq[seen].mean() / class_freq[seen]) ** power
    return w.clamp(max=max_weight)


def update_class_freq(class_freq, target_flat, momentum):
    """In-place EMA of the per-class share of Case 1 pixels with this image's
    histogram. The first update (all-zero freq, e.g. no warm start) copies it."""
    labels = target_flat[target_flat != IGNORE_LABEL]
    if labels.numel() == 0:
        return
    hist = torch.bincount(labels, minlength=NUM_CLASSES).to(class_freq.dtype)
    hist /= hist.sum()
    if class_freq.sum() == 0:
        class_freq.copy_(hist)
    else:
        class_freq.mul_(momentum).add_(hist, alpha=1.0 - momentum)


_cell_points_cache = {}


def cell_sample_points(h, w, offset, device):
    """(9, h*w) flat full-resolution pixel indices: the centre of every (h, w)
    feature cell (align_corners=True geometry, the inverse of
    lowres_index_map) and the 8 points at +-offset pixels around it, clamped
    to the image. Row 4 is the centre."""
    key = (h, w, offset, str(device))
    if key not in _cell_points_cache:
        if BB.feat_align_corners:
            cy = torch.round(torch.arange(h, device=device, dtype=torch.float32)
                             * (TARGET_H - 1) / (h - 1)).long()
            cx = torch.round(torch.arange(w, device=device, dtype=torch.float32)
                             * (TARGET_W - 1) / (w - 1)).long()
        else:
            cy = torch.round((torch.arange(h, device=device, dtype=torch.float32) + 0.5)
                             * TARGET_H / h - 0.5).long().clamp(0, TARGET_H - 1)
            cx = torch.round((torch.arange(w, device=device, dtype=torch.float32) + 0.5)
                             * TARGET_W / w - 0.5).long().clamp(0, TARGET_W - 1)
        points = []
        for dy in (-offset, 0, offset):
            for dx in (-offset, 0, offset):
                yy = (cy + dy).clamp(0, TARGET_H - 1)
                xx = (cx + dx).clamp(0, TARGET_W - 1)
                points.append((yy[:, None] * TARGET_W + xx[None, :]).reshape(-1))
        _cell_points_cache[key] = torch.stack(points)
    return _cell_points_cache[key]


def cell_labels(target_flat, h, w, offset):
    """Hard label of every target feature cell from a full-resolution label map: the
    label at the cell centre, kept only if the 8 points at +-offset px carry
    the same label (the cell lies inside one labelled region); IGNORE_LABEL
    otherwise. Gather only, so it is deterministic."""
    samples = target_flat[cell_sample_points(h, w, offset, target_flat.device)]     # (9, h*w)
    centre = samples[4]
    pure = (samples == centre[None]).all(dim=0)
    return torch.where(pure, centre, torch.full_like(centre, IGNORE_LABEL))


def proto_contrast_loss(ft_flat, cell_label, proto_bank, temperature, class_weights):
    """IDEA 1 -- prototype contrastive loss in the target feature space
    (DeepLabV2 layer4 / MiT-B5 fuse_layer), on every cell with a hard label:
        L = CE(softmax_k(cos(z, P_k) / temperature), y)
    z = the cell's target feature WITH gradient, P_k = the target prototypes
    (detached, EMA-updated as before). The own-class prototype is the
    positive and every other initialised class prototype a negative, so the
    features are pulled together per class and pushed apart between classes
    (the old L_proto / L_compress only pulled). Class-weighted like
    L_CE_agree. Returns (loss or None, number of cells, share of cells whose
    nearest prototype is their label)."""
    ready = proto_bank.target_counts > 0
    labelled = cell_label != IGNORE_LABEL
    keep = labelled & ready[cell_label.clamp(max=NUM_CLASSES - 1)]
    cells = torch.nonzero(keep, as_tuple=True)[0]
    if cells.numel() == 0 or int(ready.sum()) < 2:
        return None, 0, float('nan')
    target = cell_label[cells]
    z = F.normalize(ft_flat[:, cells], dim=0)                                  # (C, n), grad
    protos = F.normalize(proto_bank.target_protos.detach(), dim=1)             # (19, C)
    logits = (protos @ z) / temperature                                        # (19, n)
    logits = logits.masked_fill(~ready[:, None], float('-inf'))
    nll = -F.log_softmax(logits, dim=0).gather(0, target[None])[0]
    w = class_weights[target]
    acc = (logits.argmax(dim=0) == target).float().mean().item()
    return (w * nll).sum() / w.sum(), int(cells.numel()), acc


def offdiag_cos(protos, counts):
    """Mean cosine between the prototypes of different classes (lower = the
    classes are further apart in that feature space)."""
    p = F.normalize(protos[counts > 0], dim=1)
    n = p.shape[0]
    if n < 2:
        return float('nan')
    sim = p @ p.T
    return float((sim.sum() - sim.diag().sum()) / (n * (n - 1)))


def train_one_epoch(epoch, model, feat_cache, optimizer, proto_bank, text_embed,
                    entries, thresholds, args, device, csv_writer, tb_writer, class_freq, text_z,
                    ref_model=None, ev=None):
    """class_freq: (NUM_CLASSES,) running Case 1 class share, updated in place.
    text_z: text z-score statistics for the Case 1 verification (None = off).
    ref_model / ev: frozen Stage 1 model and VLM evidence (IDEA 2, None = off)."""
    rng = np.random.default_rng(args.seed + epoch)
    order = rng.permutation(len(entries))
    rows_out = []

    pbar = tqdm(order, desc=f'epoch{epoch:02d}')
    for i, idx in enumerate(pbar):
        entry = entries[idx]
        maps = compute_image_maps(model, feat_cache, entry, text_embed, device, ref_model)
        if maps is None:
            continue

        regions, _ = get_region_indices(maps['instance_mask_flat'], maps['valid_flat'],
                                        args.min_region_size)

        ce_target_flat = torch.full_like(maps['reference_flat'], IGNORE_LABEL)
        sup_mask_flat = torch.zeros_like(maps['valid_flat'])     # pixels with any supervision
        n_case1_dropped = px_case1_dropped = 0
        disagree_terms, compress_terms, consist_terms, partial_terms = [], [], [], []
        n_case1 = n_case2_high = n_case2_low = n_case2_noproto = n_case3 = 0
        n_case1_partial = px_case1_partial = 0
        n_blind = px_blind = 0
        n_case1_textrelabel = px_case1_textrelabel = 0
        px_case1 = px_case2 = 0
        q_t_vals, q_c_vals = [], []
        s_s_vals, s_c_vals, delta_vals, tilde_alpha_vals = [], [], [], []

        for region_idx in regions:
            sample_idx = sample_region(region_idx, args.max_region_sample_pixels, rng, device)
            rs = region_stats(sample_idx, maps, region_idx, args.qt_temperature)
            y_s, y_c = rs['y_s'], rs['y_c']
            q_t, q_c = rs['q_t'], rs['q_c']
            q_t_vals.append(q_t)
            q_c_vals.append(q_c)

            # ── (C) VLM-blind class: the source pseudo-label is the only
            # witness, so it is checked before the Case 1/2 decision.
            trust = blind_class_trust(sample_idx, region_idx, maps, rs, args.blind_class_ids, args)
            if trust is not None:
                k, _, pix = trust
                ce_target_flat[pix] = k
                sup_mask_flat[pix] = True
                n_blind += 1
                px_blind += pix.numel()
                continue

            # ── CASE 1 ─────────────────────────────────────────────────────
            if y_s == y_c:
                pix = case1_pixels(maps, region_idx, y_s, rs['agreement_strength'], thresholds, args)
                if pix is not None:
                    with torch.no_grad():
                        fc_region = region_feature(maps['fc_flat'], maps['fc_lowres_map'], pix)
                    rejecting = text_verify(fc_region, y_s, text_embed, text_z)
                    relabel_to = text_relabel_target(fc_region, y_s, rejecting, text_embed, text_z)
                    if relabel_to is not None:
                        # Both models agree on y_s but CLIP text strongly prefers a
                        # confuser: hard label of the confuser, no prototype update.
                        ce_target_flat[pix] = relabel_to
                        sup_mask_flat[pix] = True
                        n_case1_textrelabel += 1
                        px_case1_textrelabel += pix.numel()
                        continue
                    if rejecting:
                        # Both models agree on y_s but CLIP text prefers a confuser:
                        # partial label {y_s} + rejecting confusers, no prototype update.
                        cand = torch.tensor([y_s] + rejecting, device=device)
                        Pt_pix = maps['Pt_probs_flat'][:, pix]                       # (19, N), grad
                        partial_terms.append(-torch.log(Pt_pix[cand].sum(dim=0) + EPS))
                        n_case1_partial += 1
                        px_case1_partial += pix.numel()
                        sup_mask_flat[pix] = True
                        continue
                    if ev is not None and case1_dropped(fc_region, y_s, ev):
                        # IDEA 2(b): too little VLM-prototype evidence for y_s. No hard
                        # label, no prototype update; the pixels fall to the relabel step.
                        n_case1_dropped += 1
                        px_case1_dropped += pix.numel()
                        continue
                    ce_target_flat[pix] = y_s
                    sup_mask_flat[pix] = True
                    n_case1 += 1
                    px_case1 += pix.numel()
                    with torch.no_grad():
                        ft_region = region_feature(maps['ft_flat'].detach(), maps['ft_lowres_map'], pix)
                    proto_bank.update_target(ft_region, y_s)
                    proto_bank.update_clip(fc_region, y_c)
                else:
                    n_case3 += 1
                continue

            if not (max(q_t, q_c) > thresholds['theta_rel']
                    and abs(q_t - q_c) > thresholds['theta_margin']):
                n_case3 += 1
                continue

            # ── CASE 2 candidate: prototype compatibility ──────────────────
            with torch.no_grad():
                ft_region = region_feature(maps['ft_flat'].detach(), maps['ft_lowres_map'], region_idx)
                fc_region = region_feature(maps['fc_flat'], maps['fc_lowres_map'], region_idx)
            s_s, s_c = proto_bank.compute_compatibility(ft_region, fc_region, y_s, y_c)
            if s_s is None:
                # y_s or y_c has no prototype yet: no feature scoring. Arbitrate
                # with Stage 1's reliability-only alpha, as low-confidence Case 2
                # and without compression (there is no prototype to compress to).
                tilde_alpha = q_t / (q_t + q_c + EPS)
                tau, include_compress = TAU_LOW, False
                n_case2_low += 1
                n_case2_noproto += 1
            else:
                delta = proto_bank.compute_delta(s_s, s_c)
                tilde_qt, tilde_qc = q_t * s_s, q_c * s_c
                tilde_alpha = tilde_qt / (tilde_qt + tilde_qc + EPS)
                s_s_vals.append(s_s)
                s_c_vals.append(s_c)
                delta_vals.append(delta)
                tilde_alpha_vals.append(tilde_alpha)

                if delta > thresholds['theta_feat'] and max(s_s, s_c) > thresholds['theta_compat']:
                    tau, include_compress = TAU_HIGH, True
                    n_case2_high += 1
                elif delta > thresholds['theta_feat'] / 2:
                    tau, include_compress = TAU_LOW, False
                    n_case2_low += 1
                else:
                    n_case3 += 1          # demoted
                    continue
            px_case2 += region_idx.numel()
            sup_mask_flat[region_idx] = True

            teacher = rs['Pt_mean'] if tilde_alpha > 0.5 else rs['Pc_mean']
            P_hat = F.softmax(torch.log(teacher + EPS) / tau, dim=0).detach()
            y_hat = int(P_hat.argmax())

            Pt_region = maps['Pt_probs_flat'][:, region_idx]                       # (19, N), grad
            disagree_terms.append(-(P_hat[:, None] * torch.log(Pt_region + EPS)).sum(dim=0))

            # Skip compression if y_hat has no prototype yet: a zero prototype
            # would just pull the region's features towards the origin.
            # Both sides unit-norm, so each term is 2 - 2*cos and lies in [0, 4].
            if include_compress and proto_bank.target_counts[y_hat] > 0:
                ft_region_grad = F.normalize(
                    region_feature(maps['ft_flat'], maps['ft_lowres_map'], region_idx), dim=0)
                proto = proto_bank.get_target_proto(y_hat)
                compress_terms.append(torch.norm(ft_region_grad - proto) ** 2)

            # KL towards VLM only when VLM is the selected teacher;
            # otherwise it would fight L_CE_disagree's pull towards Pt.
            if tilde_alpha <= 0.5:
                Pt_mean_grad = Pt_region.mean(dim=1)
                Pc_mean = rs['Pc_mean'].detach()
                consist_terms.append((Pc_mean * torch.log(Pc_mean / (Pt_mean_grad + EPS) + EPS)).sum())

        # ── Losses ─────────────────────────────────────────────────────────
        zero = torch.tensor(0., device=device)
        # Class-balanced Case 1 CE: weights from the class share BEFORE this
        # image, then the share is updated with this image's Case 1 pixels.
        logits_flat = maps['logits'][0].reshape(NUM_CLASSES, -1)
        cb_weights = class_balance_weights(class_freq, args.cb_power, args.cb_max_weight)
        L_CE_agree = masked_ce(logits_flat, ce_target_flat, cb_weights)
        L_CE_agree = zero if L_CE_agree is None else L_CE_agree
        with torch.no_grad():
            L_CE_agree_unw = masked_ce(logits_flat, ce_target_flat)          # logged only
        update_class_freq(class_freq, ce_target_flat, args.cb_momentum)
        L_CE_disagree = torch.cat(disagree_terms).mean() if disagree_terms else zero
        L_compress = torch.stack(compress_terms).mean() if compress_terms else zero
        L_consist = torch.stack(consist_terms).mean() if consist_terms else zero
        L_partial = torch.cat(partial_terms).mean() if partial_terms else zero
        # IDEA 2(a): prototype-rectified labels on the pixels nothing else supervises.
        L_CE_relabel, px_relabel, px_relabel_changed = zero, 0, 0
        if ev is not None and math.isfinite(ev['relabel_threshold']):
            relabel_target_flat = torch.full_like(ce_target_flat, IGNORE_LABEL)
            with torch.no_grad():
                for pix, label, conf, stage_c_label in unsupervised_units(maps, sup_mask_flat, ev):
                    if conf >= ev['relabel_threshold']:
                        relabel_target_flat[pix] = label
                        px_relabel += pix.numel()
                        px_relabel_changed += pix.numel() if label != stage_c_label else 0
            L_CE_relabel = masked_ce(logits_flat, relabel_target_flat, cb_weights)
            L_CE_relabel = zero if L_CE_relabel is None else L_CE_relabel
        valid_flat = maps['valid_flat']
        if valid_flat.any():
            mean_pred = maps['Pt_probs_flat'][:, valid_flat].mean(dim=1)
            L_balance = (mean_pred * torch.log(mean_pred + EPS)).sum()
        else:
            L_balance = zero

        # IDEA 1: prototype contrastive loss on the target feature cells inside Case 1 labels.
        L_pc, pc_cells, pc_acc = None, 0, float('nan')
        if args.proto_contrast == 'on':
            h4, w4 = maps['ft_hw']
            L_pc, pc_cells, pc_acc = proto_contrast_loss(
                maps['ft_flat'], cell_labels(ce_target_flat, h4, w4, args.pc_cell_offset),
                proto_bank, args.pc_temperature, cb_weights)
        L_pc = zero if L_pc is None else L_pc

        L_total = (L_CE_agree
                   + args.lambda_pc * L_pc
                   + args.lambda_relabel * L_CE_relabel
                   + args.lambda_dis * L_CE_disagree
                   + args.lambda_comp * L_compress
                   + args.lambda_cons * L_consist
                   + args.lambda_partial * L_partial
                   + args.mu_balance * L_balance)

        optimizer.zero_grad()
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ── Logging ────────────────────────────────────────────────────────
        n_valid = int(valid_flat.sum())
        px_case3 = n_valid - px_case1 - px_case2 - px_case1_partial - px_blind - px_case1_textrelabel
        row = {
            'epoch': epoch, 'image_idx': int(idx), 'city': entry['city'], 'stem': entry['stem'],
            'n_case1': n_case1, 'n_case2_high': n_case2_high, 'n_case2_low': n_case2_low,
            'n_case3': n_case3, 'n_case2_noproto': n_case2_noproto,
            'n_case1_partial': n_case1_partial, 'px_case1_partial': px_case1_partial,
            'L_partial': L_partial.item(),
            'px_case1': px_case1, 'px_case2': px_case2, 'px_case3': px_case3,
            'ignored_frac': px_case3 / max(1, n_valid),
            'mean_q_t': nanmean(q_t_vals), 'mean_q_c': nanmean(q_c_vals),
            'mean_s_s': nanmean(s_s_vals), 'mean_s_c': nanmean(s_c_vals),
            'mean_delta': nanmean(delta_vals), 'mean_tilde_alpha': nanmean(tilde_alpha_vals),
            'L_CE_agree': L_CE_agree.item(),
            'L_CE_agree_unweighted': 0.0 if L_CE_agree_unw is None else L_CE_agree_unw.item(),
            'cb_weight_max': float(cb_weights[class_freq > 0].max()) if (class_freq > 0).any() else 1.0,
            'L_CE_disagree': L_CE_disagree.item(),
            'L_compress': L_compress.item(), 'L_consist': L_consist.item(),
            'L_balance': L_balance.item(), 'L_total': L_total.item(),
            'L_proto_contrast': L_pc.item(), 'pc_cells': pc_cells, 'pc_acc': pc_acc,
            'L_CE_relabel': L_CE_relabel.item(), 'px_relabel': px_relabel,
            'px_relabel_changed': px_relabel_changed,
            'n_case1_dropped': n_case1_dropped, 'px_case1_dropped': px_case1_dropped,
            'n_blind': n_blind, 'px_blind': px_blind,
            'n_case1_textrelabel': n_case1_textrelabel, 'px_case1_textrelabel': px_case1_textrelabel,
        }
        csv_writer.writerow(row)
        rows_out.append(row)
        global_step = (epoch - 1) * len(entries) + i
        for key in TB_ITER_FIELDS:
            if not (isinstance(row[key], float) and math.isnan(row[key])):
                tb_writer.add_scalar(f'train_iter/{key}', row[key], global_step)
        pbar.set_postfix(agree=f'{row["L_CE_agree"]:.3f}', dis=f'{row["L_CE_disagree"]:.3f}',
                         pc=f'{row["L_proto_contrast"]:.3f}', ign=f'{row["ignored_frac"]:.3f}')

        if i < 5:
            print(f'\n[diag epoch{epoch} img{i}] case1={n_case1} case1_partial={n_case1_partial} case2_high={n_case2_high} '
                  f'case2_low={n_case2_low} (noproto={n_case2_noproto}) case3={n_case3} ignored_frac={row["ignored_frac"]:.3f} '
                  f'proto_ready={proto_bank.is_ready()}')
        if len(rows_out) % 50 == 0:
            torch.cuda.empty_cache()

    return rows_out


def epoch_summary(rows):
    summary = {}
    for key in CSV_FIELDS:
        if key in ('epoch', 'image_idx', 'city', 'stem'):
            continue
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
        summary[key] = float(np.mean(vals)) if vals else float('nan')
    return summary


def main():
    parser = argparse.ArgumentParser(description='ReFuSe Stage 2')
    parser.add_argument('--config', required=True, help='Experiment config (configs/*.yaml)')
    parser.add_argument('--ref-dir', default=None,
                        help='Where the frozen Stage 1 reference labels are written / reused '
                             '(default <work_dir>/stage1_reference_labels/train)')
    parser.add_argument('--stage-c-ckpt', default=None,
                        help='Stage 1 best model: the frozen reference in every epoch, and the '
                             'initial weights (default <work_dir>/stage1/best.pth)')
    parser.add_argument('--init-from-source', action='store_true',
                        help='Initialise the trained model from the source model instead of Stage 1 '
                             '(ablation); the reference stays frozen Stage 1')
    parser.add_argument('--proto-ckpt', default=None, help='Optional pre-built prototype bank')
    parser.add_argument('--viz-count', type=int, default=10,
                        help='Evenly spaced held-out val images rendered as image | prediction '
                             'panels after every epoch (plus epoch00 = initial model) under '
                             'train_qualitative_viz/. Diagnostic only. 0 disables.')
    parser.add_argument('--holdout-images', type=int, default=200,
                        help='Train images held out (per city, --seed) as the pseudo-label val '
                             'set for best.pth and early stopping; never trained on. Capped at '
                             'half the images (for --debug).')
    parser.add_argument('--val-pseudo-label', choices=VAL_REFERENCES, default='consensus',
                        help="Pseudo-label that val mIoU (best.pth) and val loss (early stopping) "
                             "use: 'consensus' = Ps where Ps == Pc (both teachers agree), 'pc' = "
                             "VLM, 'ps' = source pseudo-label. All three are logged.")
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='Evaluate on the held-out val images every N epochs (and at '
                             '--max-epochs). Only evaluated epochs can become best.pth.')
    parser.add_argument('--ckpt-interval', type=int, default=5,
                        help='Save dsp_stage_epochNN.pth only at epoch 1 and every N-th epoch '
                             '(1, 5, 10, 15, ...). best.pth is saved on every new best regardless. '
                             'Only these epochs can be passed to --resume.')
    parser.add_argument('--resume', default=None,
                        help='Resume from a dsp_stage_epochNN.pth of this script '
                             '(model, optimizer, prototypes, thresholds, history)')
    parser.add_argument('--work-dir', default=None, help='Overrides work_dir of the config')
    parser.add_argument('--max-epochs', type=int, default=30,
                        help='Upper bound; training may stop earlier (--early-stop-patience)')
    parser.add_argument('--early-stop-patience', type=int, default=5,
                        help='Stop after this many consecutive evaluations (not epochs; see '
                             '--eval-interval) without the val loss dropping '
                             '(0 = disabled, always run to --max-epochs)')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0,
                        help='An evaluation counts as an improvement only if the val loss drops '
                             'by more than this')
    parser.add_argument('--lr', type=float, default=None,
                        help='Adam learning rate (default: 1e-5 for deeplabv2, 6e-5 for mitb5)')
    parser.add_argument('--lambda-dis', type=float, default=1.0)
    parser.add_argument('--lambda-comp', type=float, default=0.1)
    parser.add_argument('--lambda-cons', type=float, default=0.1)
    parser.add_argument('--mu-balance', type=float, default=0.01)
    parser.add_argument('--cb-power', type=float, default=0.5,
                        help='Class-balanced L_CE_agree: w_c = (mean f / f_c) ** power over the running '
                             'Case 1 class share f (0.5 = sqrt inverse frequency; 0 = unweighted)')
    parser.add_argument('--cb-max-weight', type=float, default=3.0,
                        help='Upper clamp of the class-balance weights')
    parser.add_argument('--text-verify', dest='text_verify', action='store_true',
                        help='Verify Case 1 labels in --text-verify-pairs with z-scored CLIP text '
                             'similarity; rejected regions get the partial label (default on)')
    parser.add_argument('--no-text-verify', dest='text_verify', action='store_false')
    parser.set_defaults(text_verify=True)
    parser.add_argument('--text-verify-pairs', default='auto',
                        help="'auto' (default): every rare Case 1 label (pixel share within "
                             "--text-z-classes < --text-verify-max-share) is checked against its "
                             "--text-verify-topk nearest classes in the CLIP text embedding -- no "
                             "hand-picked pairs, recomputed for whichever CLIP model is used. Or "
                             "comma-separated 'label:confuser' pairs, e.g. 'bus:train'. A Case 1 "
                             "region labelled 'label' is rejected if z(confuser) - z(label) > "
                             "--text-verify-margin.")
    parser.add_argument('--text-verify-topk', type=int, default=2,
                        help="'auto' pairs: number of nearest text-embedding classes per label")
    parser.add_argument('--text-verify-max-share', type=float, default=0.01,
                        help="'auto' pairs: only labels below this Case 1 pixel share are checked")
    parser.add_argument('--text-verify-action', choices=['relabel', 'partial'], default='relabel',
                        help="A text-rejected Case 1 region: hard label of the strongest confuser if "
                             "z(c) - z(y) > --text-relabel-margin ('relabel'), or always the partial "
                             "label ('partial')")
    parser.add_argument('--text-relabel-margin', type=float, default=1.0,
                        help='z-score gap above which a rejected Case 1 region is relabelled')
    parser.add_argument('--text-verify-margin', type=float, default=0.5,
                        help='Reject only if z(confuser) exceeds z(label) by more than this')
    parser.add_argument('--text-z-classes', default=None,
                        help='Case 1 labels whose warm-start regions define the per-class text '
                             'cosine mean / std (default: building, wall, fence, person, rider, car, '
                             'truck, bus, train, motorcycle, bicycle minus the source-absent classes)')
    parser.add_argument('--lambda-partial', type=float, default=1.0,
                        help='Weight of L_partial (-log sum of Pt over {label} + rejecting confusers)')
    parser.add_argument('--cb-momentum', type=float, default=0.99,
                        help='EMA momentum of the running Case 1 class share (per image)')
    parser.add_argument('--min-region-size', type=int, default=50)
    parser.add_argument('--theta-agree', type=float, default=0.0,
                        help='Case 1 needs agreement_strength > theta_agree. Default 0.0 = no '
                             'condition beyond y_s == y_c (with --case1-pixel-agree only the '
                             'agreeing pixels are labelled). Negative = use the calibrated '
                             '60th percentile.')
    parser.add_argument('--case1-pixel-agree', dest='case1_pixel_agree', action='store_true',
                        help='Case 1: label (and pool prototypes over) only pixels where the '
                             'reference and Pc_label both equal y_s (default on)')
    parser.add_argument('--no-case1-pixel-agree', dest='case1_pixel_agree', action='store_false',
                        help='Case 1: label the whole region')
    parser.set_defaults(case1_pixel_agree=True)
    parser.add_argument('--calib-images', type=int, default=50)
    parser.add_argument('--force-recalibrate', action='store_true')
    parser.add_argument('--qt-temperature', type=float, default=2.0,
                        help="Softening of Pt before q_t's entropy term (Stage 1); 1 disables")
    parser.add_argument('--max-region-sample-pixels', type=int, default=5000,
                        help='Pixel cap for region statistics; losses always use the full region')
    parser.add_argument('--proto-momentum', type=float, default=0.99)
    parser.add_argument('--proto-contrast', choices=['off', 'on'], default='off',
                        help="IDEA 1. 'on': prototype contrastive loss in the target feature space "
                             'on every cell inside a Case 1 label (see proto_contrast_loss)')
    parser.add_argument('--lambda-pc', type=float, default=0.1, help='Weight of the prototype contrastive loss')
    parser.add_argument('--pc-temperature', type=float, default=0.1,
                        help='Temperature of the prototype contrastive softmax over cosines')
    parser.add_argument('--pc-cell-offset', type=int, default=None,
                        help='A target feature cell gets a label only if its centre and the 8 points '
                             'at +-this many pixels carry the same Case 1 label (default: 4 for '
                             'deeplabv2 layer4 cells ~16 px apart, 2 for mitb5 cells ~8 px apart)')
    parser.add_argument('--proto-denoise', choices=['off', 'on'], default='off',
                        help="IDEA 2. 'on': frozen VLM visual prototypes give pseudo-label "
                             'evidence w: (a) the unsupervised pixels get argmax(P_stageC * w) as a '
                             'hard label where it is confident (--denoise-coverage), (b) Case 1 '
                             'regions with the least evidence for their label lose it '
                             '(--denoise-c1-drop)')
    parser.add_argument('--denoise-temperature', type=float, default=0.02,
                        help='Temperature of w = softmax(cos(VLM feature, prototype) / T)')
    parser.add_argument('--denoise-coverage', type=float, default=0.5,
                        help='IDEA 2(a): share of the unsupervised pixels (most confident units '
                             'first, calibrated on --denoise-calib-images) that get a label. 0 disables.')
    parser.add_argument('--denoise-c1-drop', type=float, default=0.0,
                        help='IDEA 2(b): per Case 1 label, the pixel share with the lowest prototype '
                             'evidence that loses its hard label (calibrated on the warm start). '
                             'Default 0 = off: in the audit a 0.1 drop removed 28.5%% of the wrong '
                             'Case 1 pixels but also 7.7%% of the correct ones (2.2 correct pixels per '
                             'wrong one), no better than Stage 1 confidence. Try 0.1 as a sub-ablation.')
    parser.add_argument('--denoise-min-regions', type=int, default=20,
                        help='A class needs this many warm-start Case 1 regions for a VLM '
                             'evidence prototype; otherwise it is neutral in w and never dropped')
    parser.add_argument('--denoise-calib-images', type=int, default=100)
    parser.add_argument('--lambda-relabel', type=float, default=0.5,
                        help='Weight of the CE on the IDEA 2(a) labels')
    parser.add_argument('--blind-classes', default='auto',
                        help="VLM-blind classes of rule (C): 'auto' = the ones stored in --stage-c-ckpt "
                             "(none if it has no 'blind_classes'), 'none', or comma-separated class names")
    parser.add_argument('--blind-q-thresh', type=float, default=0.6,
                        help='q_src above which a blind-class region gets its hard label (Stage 1 value)')
    parser.add_argument('--run-name', default=None,
                        help='Output subdir under <work_dir>; default '
                             'stage2_contrast<on|off>_denoise<on|off>')
    parser.add_argument('--warmstart-images', type=int, default=500,
                        help='Training images used to pre-build prototypes from Case 1 regions '
                             'before epoch 1 (no gradient). 0 disables, -1 uses all. Skipped '
                             'when resuming or when --proto-ckpt is given.')
    parser.add_argument('--seed', type=int, default=42)
    # --deterministic / --no-deterministic (argparse.BooleanOptionalAction needs Python >= 3.9)
    parser.add_argument('--deterministic', dest='deterministic', action='store_true',
                        help='Deterministic cuDNN/cuBLAS kernels so a rerun with the same seed '
                             'reproduces the run (default on; --no-deterministic is faster)')
    parser.add_argument('--no-deterministic', dest='deterministic', action='store_false')
    parser.set_defaults(deterministic=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--debug', action='store_true', help='First 2 train images per city')
    args = parser.parse_args()

    global BB, VLM_T, ABSENT_CLASSES, ABSENT_T
    cfg = load_config(args.config)
    BB = make_backbone(cfg)
    ABSENT_CLASSES = cfg['absent_classes']
    args.work_dir = args.work_dir or cfg['work_dir']
    args.img_dir = cfg['data']['train_img_dir']
    args.ps_dir = cfg['data']['ps_root']
    args.sam2_dir = cfg['data']['sam2_root']
    args.source_ckpt = cfg['source_ckpt']
    args.stage_c_ckpt = args.stage_c_ckpt or os.path.join(args.work_dir, 'stage1', 'best.pth')
    args.ref_dir = args.ref_dir or os.path.join(args.work_dir, 'stage1_reference_labels', 'train')
    if args.lr is None:
        args.lr = BB.default_lr
    if args.pc_cell_offset is None:
        args.pc_cell_offset = BB.default_pc_cell_offset
    if args.text_z_classes is None:
        args.text_z_classes = default_text_z_classes(ABSENT_CLASSES)
    print(f"Config {cfg['config_path']}: source={cfg['source']} backbone={BB.name} vlm={cfg['vlm']['name']} "
          f"lr={args.lr} pc_cell_offset={args.pc_cell_offset}")
    # Parsed forms (lists/dicts, so vars(args) stays JSON-serialisable for run_config.json).
    if args.text_verify_pairs != 'auto':
        parse_verify_pairs(args.text_verify_pairs)        # fail early on a bad class name
    args.text_z_class_ids = sorted(parse_class_list(args.text_z_classes))
    if args.blind_classes == 'auto':
        stage_c = torch.load(args.stage_c_ckpt, map_location='cpu')
        args.blind_class_ids = sorted(int(k) for k in stage_c.get('blind_classes', [])) \
            if isinstance(stage_c, dict) else []
        del stage_c
    else:
        args.blind_class_ids = [] if args.blind_classes == 'none' else sorted(parse_class_list(args.blind_classes))
    print(f'VLM-blind classes ({args.blind_classes}): '
          + (', '.join(CITYSCAPES_CLASSES[k] for k in args.blind_class_ids) or 'none'))

    setup_reproducibility(args.seed, args.deterministic)
    device = torch.device(f'cuda:{args.gpu}')
    ABSENT_T = torch.tensor(ABSENT_CLASSES, dtype=torch.long, device=device)
    VLM_T = VLM.create(cfg['vlm'], device, ABSENT_CLASSES)

    run_name = args.run_name or f'stage2_contrast{args.proto_contrast}_denoise{args.proto_denoise}'
    ckpt_dir = os.path.join(args.work_dir, run_name)
    print(f'Run: proto_contrast={args.proto_contrast} proto_denoise={args.proto_denoise} -> {ckpt_dir}')
    os.makedirs(ckpt_dir, exist_ok=True)
    thresholds_path = os.path.join(ckpt_dir, 'thresholds.json')
    csv_path = os.path.join(ckpt_dir, 'train_log.csv')
    summary_path = os.path.join(ckpt_dir, 'epoch_summary.jsonl')

    # ── Model ──────────────────────────────────────────────────────────────
    model = BB.build(device)                       # source-only weights
    resume = torch.load(args.resume, map_location='cpu') if args.resume else None
    if resume is not None:
        model.load_state_dict(resume['model'], strict=True)
        print(f'Resuming from {args.resume} (epoch {resume["epoch"]})')
        for key in ('proto_contrast', 'proto_denoise'):
            if resume['args'].get(key, 'off') != getattr(args, key):
                raise ValueError(f"{args.resume} was trained with --{key.replace('_', '-')} "
                                 f"{resume['args'].get(key, 'off')}; pass the same switch.")
    else:
        init_path = args.source_ckpt if args.init_from_source else args.stage_c_ckpt
        if not args.init_from_source:
            model.load_state_dict(load_model_weights(init_path), strict=True)
        print(f'Initialised from {init_path}')
    model.to(device)
    model.train()
    freeze_bn(model)

    feat_cache = {}

    def _hook(module, inp, out):
        feat_cache['ft'] = out
    hook = BB.feature_layer(model).register_forward_hook(_hook)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer'])

    proto_bank = DualPrototypeBank(num_classes=NUM_CLASSES, target_feat_dim=BB.feat_dim,
                                   clip_feat_dim=VLM_T.text_embed.shape[1],
                                   ema_momentum=args.proto_momentum, device=str(device))
    if resume is not None:
        proto_bank.load_state_dict(resume['proto_bank'])
    elif args.proto_ckpt:
        proto_bank.load(args.proto_ckpt)
        print(f'Loaded prototypes from {args.proto_ckpt}')

    # IDEA 2: the frozen Stage 1 model supplies P_stageC for the rectified labels.
    ref_model = None
    if args.proto_denoise == 'on':
        ref_model = BB.build(device, args.stage_c_ckpt).eval()
        for prm in ref_model.parameters():
            prm.requires_grad_(False)

    text_embed = VLM_T.text_embed                  # (19, 512), L2-normalised

    entries = collect_entries(args)
    if args.debug:
        seen = defaultdict(int)
        kept = []
        for e in entries:
            if seen[e['city']] < 2:
                kept.append(e)
                seen[e['city']] += 1
        entries = kept
    n_holdout = min(args.holdout_images, len(entries) // 2)
    entries, holdout_entries = stratified_holdout_split(entries, n_holdout, seed=args.seed)
    ensure_reference_labels(entries, args.stage_c_ckpt, args.ref_dir, device)
    holdout_stems = sorted(e['stem'] for e in holdout_entries)
    if resume is not None and resume.get('holdout_stems') != holdout_stems:
        raise ValueError(f'{args.resume} was trained with a different held-out split (or by a script '
                         'without one); use the same --holdout-images/--seed/--debug, or start fresh.')
    print(f'{len(entries)} train images, {len(holdout_entries)} held out as pseudo-label val '
          f"(best.pth / early stopping vs '{args.val_pseudo_label}'); Cityscapes val is not used")

    # ── Thresholds ─────────────────────────────────────────────────────────
    # No-grad passes (calibration, warm start, relabel calibration) in eval mode:
    # no dropout / stochastic depth (MiT-B5). Back to train mode before training.
    model.eval()
    if resume is not None:
        thresholds = resume['thresholds']
    else:
        thresholds = None
        calib_config = calibration_config(args, init_path)
        if os.path.exists(thresholds_path) and not args.force_recalibrate:
            with open(thresholds_path) as f:
                cached = json.load(f)
            cached_config = cached.get('calibration_config', {})
            if cached_config == calib_config:
                thresholds = cached
                print(f'Loaded cached thresholds from {thresholds_path}')
            else:
                changed = sorted(k for k in set(calib_config) | set(cached_config)
                                 if calib_config.get(k) != cached_config.get(k))
                print(f'Cached {thresholds_path} was calibrated with different settings '
                      f'({", ".join(changed)}) -- recalibrating')
    if thresholds is None:
        calib_rng = np.random.default_rng(args.seed)
        n_calib = min(args.calib_images, len(entries))
        calib_entries = [entries[j] for j in calib_rng.choice(len(entries), n_calib, replace=False)]
        thresholds = calibrate_thresholds(model, feat_cache, calib_entries, text_embed, args, device)
        thresholds['calibration_config'] = calib_config
        with open(thresholds_path, 'w') as f:
            json.dump(thresholds, f, indent=2)
        print(f'Saved thresholds -> {thresholds_path}')
    # thresholds.json keeps the calibrated theta_agree; the run uses --theta-agree.
    if 'theta_agree_calibrated' not in thresholds:
        thresholds['theta_agree_calibrated'] = thresholds['theta_agree']
    thresholds['theta_agree'] = (thresholds['theta_agree_calibrated'] if args.theta_agree < 0
                                 else args.theta_agree)
    print('Thresholds: ' + '  '.join(f'{k}={v:.4f}' for k, v in thresholds.items() if k.startswith('theta')))

    if resume is None and not args.proto_ckpt and args.warmstart_images != 0:
        warm_rng = np.random.default_rng(args.seed + 2)
        n_warm = len(entries) if args.warmstart_images < 0 else min(args.warmstart_images, len(entries))
        warm_entries = [entries[j] for j in warm_rng.choice(len(entries), n_warm, replace=False)]
        class_freq, text_z, ev = warmstart_prototypes(model, feat_cache, proto_bank, warm_entries,
                                                      text_embed, thresholds, args, device)
        proto_bank.save(os.path.join(ckpt_dir, 'proto_bank_warmstart.pth'))
        if ev is not None and args.denoise_coverage > 0:
            n_cal = min(args.denoise_calib_images, len(warm_entries))
            ev['relabel_threshold'] = calibrate_relabel_threshold(
                model, feat_cache, ref_model, warm_entries[:n_cal], text_embed, thresholds,
                text_z, ev, args, device)
    elif resume is not None:
        class_freq = resume['case1_class_freq'].float().to(device)
        ev = resume.get('clip_evidence')
        if ev is not None:
            ev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in ev.items()}
        text_z = resume.get('text_z')
        if text_z is not None:
            text_z = {k: v.to(device) if torch.is_tensor(v) else v for k, v in text_z.items()}
        if args.text_verify and text_z is None:
            raise ValueError(f'{args.resume} has no text z-score statistics; pass --no-text-verify '
                             'or start fresh.')
    else:
        if args.text_verify:
            raise ValueError('--text-verify needs the warm start for its z-score statistics '
                             '(--warmstart-images != 0 and no --proto-ckpt), or pass --no-text-verify.')
        if args.proto_denoise == 'on':
            raise ValueError('--proto-denoise on needs the warm start for its VLM prototypes '
                             '(--warmstart-images != 0 and no --proto-ckpt).')
        class_freq = torch.zeros(NUM_CLASSES, device=device)   # set by the first training image
        text_z = None
        ev = None
    model.train()
    freeze_bn(model)
    if ev is not None:
        with open(os.path.join(ckpt_dir, 'clip_evidence.json'), 'w') as f:
            json.dump({'temperature': ev['temperature'], 'relabel_threshold': ev['relabel_threshold'],
                       'coverage': args.denoise_coverage, 'c1_drop': args.denoise_c1_drop,
                       'classes_with_prototype': [n for n, r in zip(CITYSCAPES_CLASSES, ev['ready'].tolist()) if r],
                       'c1_threshold': dict(zip(CITYSCAPES_CLASSES, ev['c1_threshold'].tolist()))},
                      f, indent=2)
        print(f"VLM evidence: T={ev['temperature']}  relabel threshold={ev['relabel_threshold']:.4f}  "
              'Case 1 drop thresholds: ' + '  '.join(
                  f'{n}={t:.3f}' for n, t in zip(CITYSCAPES_CLASSES, ev['c1_threshold'].tolist())
                  if math.isfinite(t)))
    if text_z is not None:
        with open(os.path.join(ckpt_dir, 'text_z_stats.json'), 'w') as f:
            json.dump({'n_regions': text_z['n_regions'],
                       'margin': text_z['margin'],
                       'relabel_margin': text_z.get('relabel_margin'),
                       'no_relabel': [CITYSCAPES_CLASSES[k] for k in text_z.get('no_relabel', ())],
                       'pairs': {CITYSCAPES_CLASSES[y]: [CITYSCAPES_CLASSES[c] for c in cs]
                                 for y, cs in sorted(text_z['pairs'].items())},
                       'mu': dict(zip(CITYSCAPES_CLASSES, text_z['mu'].tolist())),
                       'sd': dict(zip(CITYSCAPES_CLASSES, text_z['sd'].tolist()))}, f, indent=2)
    print('Case 1 class share: ' + '  '.join(
        f'{n}={v:.4f}' for n, v in zip(CITYSCAPES_CLASSES, class_freq.tolist()) if v > 0))
    print('Initial class-balance weights: ' + '  '.join(
        f'{n}={w:.2f}' for n, w, v in zip(CITYSCAPES_CLASSES,
                                         class_balance_weights(class_freq, args.cb_power,
                                                               args.cb_max_weight).tolist(),
                                         class_freq.tolist()) if v > 0))

    # Checkpoint selection / early-stopping state: best.pth on held-out
    # pseudo-label mIoU, early stopping on held-out loss.
    if resume is not None:
        if resume['args'].get('val_pseudo_label') != args.val_pseudo_label:
            raise ValueError(f"{args.resume} selected checkpoints against "
                             f"'{resume['args'].get('val_pseudo_label')}'; pass the same --val-pseudo-label.")
        history = resume['loss_history']
        best_val_miou = resume['best_val_miou']
        best_val_loss = resume['best_val_loss']
        epochs_no_improve = resume['epochs_no_improve']
        baseline = resume['baseline_val']
    else:
        history = []
        best_val_miou, best_val_loss, epochs_no_improve = float('-inf'), float('inf'), 0
    start_epoch = resume['epoch'] + 1 if resume is not None else 1
    prepare_logs(ckpt_dir, resume['epoch'] if resume is not None else None)
    save_run_config(ckpt_dir, args, start_epoch)
    iter_writer, epoch_writer = open_tensorboard(ckpt_dir, start_epoch, len(entries))
    miou_key = f'val_miou_{args.val_pseudo_label}'

    if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
        print(f'Checkpoint already reached early stop ({epochs_no_improve} evaluations without the '
              f'val loss dropping, patience {args.early_stop_patience}). Pass a larger '
              '--early-stop-patience (or 0) to keep training.')
        iter_writer.close()
        epoch_writer.close()
        hook.remove()
        return

    viz_dir = os.path.join(ckpt_dir, 'train_qualitative_viz')
    val_sample = fixed_viz_sample(holdout_entries, args.viz_count)
    if val_sample:
        print(f'Qualitative viz: {len(val_sample)} held-out val images -> {viz_dir}/epochNN/')
    if resume is None:
        # Epoch 0 = the initial (Stage 1) model: the baseline every epoch is compared
        # against. Logged only; best.pth is always a trained epoch.
        baseline = evaluate_holdout(model, holdout_entries, device, args.val_pseudo_label)
        log_val_to_tensorboard(epoch_writer, baseline, 0)
        print(f"[epoch 0 / initial model] val mIoU vs {args.val_pseudo_label}={baseline[miou_key]:.2f}  "
              f"val_loss={baseline['val_loss']:.4f}  "
              + '  '.join(f"mIoU_{r}={baseline[f'val_miou_{r}']:.2f}" for r in VAL_REFERENCES))
        panels = save_training_viz(model, val_sample, args.img_dir, device, viz_dir, 0)
        log_viz_to_tensorboard(epoch_writer, panels, 0)

    csv_is_new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if csv_is_new:
            csv_writer.writeheader()

        for epoch in range(start_epoch, args.max_epochs + 1):
            rows = train_one_epoch(epoch, model, feat_cache, optimizer, proto_bank, text_embed,
                                   entries, thresholds, args, device, csv_writer, iter_writer,
                                   class_freq, text_z, ref_model, ev)
            csv_file.flush()

            summary = epoch_summary(rows)
            summary['epoch'] = epoch
            summary['target_proto_offdiag_cos'] = offdiag_cos(proto_bank.target_protos, proto_bank.target_counts)
            summary['clip_proto_offdiag_cos'] = offdiag_cos(proto_bank.clip_protos, proto_bank.clip_counts)
            print(f'\n[epoch {epoch}] ' + '  '.join(
                f'{k}={v:.4f}' for k, v in summary.items() if k != 'epoch'))
            proto_bank.summary()
            for k, v in summary.items():
                if k != 'epoch' and not math.isnan(v):
                    epoch_writer.add_scalar(f'train_epoch/{k}', v, epoch)

            # ── Held-out pseudo-label validation ───────────────────────────
            is_best = stop_early = False
            if epoch % args.eval_interval == 0 or epoch == args.max_epochs:
                val = evaluate_holdout(model, holdout_entries, device, args.val_pseudo_label)
                summary.update({k: v for k, v in val.items() if k.startswith('val_')})
                log_val_to_tensorboard(epoch_writer, val, epoch)

                is_best = val[miou_key] > best_val_miou
                if is_best:
                    best_val_miou = val[miou_key]
                if val['val_loss'] < best_val_loss - args.early_stop_min_delta:
                    best_val_loss = val['val_loss']
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                stop_early = args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience

                print(f"  val mIoU vs {args.val_pseudo_label}={val[miou_key]:.2f} "
                      f"(best={best_val_miou:.2f}, epoch 0={baseline[miou_key]:.2f})"
                      + (' (Best!)' if is_best else '')
                      + '  ' + '  '.join(f"mIoU_{r}={val[f'val_miou_{r}']:.2f}" for r in VAL_REFERENCES))
                print(f"  val_loss={val['val_loss']:.4f} (best={best_val_loss:.4f}, epoch 0="
                      f"{baseline['val_loss']:.4f}, no_improve={epochs_no_improve}/{args.early_stop_patience})")
                epoch_writer.add_scalar('early_stop/best_val_miou', best_val_miou, epoch)
                epoch_writer.add_scalar('early_stop/best_val_loss', best_val_loss, epoch)
                epoch_writer.add_scalar('early_stop/evals_no_improve', epochs_no_improve, epoch)

            history.append(summary)
            with open(summary_path, 'a') as f:
                f.write(json.dumps(summary) + '\n')
            epoch_writer.add_scalar('prototypes/target_classes_ready',
                                    int((proto_bank.target_counts > 0).sum()), epoch)
            epoch_writer.add_scalar('prototypes/clip_classes_ready',
                                    int((proto_bank.clip_counts > 0).sum()), epoch)
            cb_w = class_balance_weights(class_freq, args.cb_power, args.cb_max_weight)
            for name, f, w in zip(CITYSCAPES_CLASSES, class_freq.tolist(), cb_w.tolist()):
                epoch_writer.add_scalar(f'case1_class_share/{name}', f, epoch)
                epoch_writer.add_scalar(f'case1_class_weight/{name}', w, epoch)

            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'target_model': model.state_dict(),   # same key as Stage 1, for evaluate.py
                'optimizer': optimizer.state_dict(),
                'proto_bank': proto_bank.state_dict(),
                'thresholds': thresholds,
                'loss_history': history,
                'best_val_miou': best_val_miou,
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve,
                'baseline_val': baseline,
                'holdout_stems': holdout_stems,
                'case1_class_freq': class_freq.cpu(),
                'clip_evidence': (None if ev is None else
                                  {k: v.cpu() if torch.is_tensor(v) else v for k, v in ev.items()}),
                'text_z': (None if text_z is None else
                           {k: v.cpu() if torch.is_tensor(v) else v for k, v in text_z.items()}),
                'args': vars(args),
            }
            # The final epoch (early stop or --max-epochs) is always saved, so a
            # finished run can be resumed with a larger patience / --max-epochs.
            if (epoch == 1 or epoch % args.ckpt_interval == 0
                    or stop_early or epoch == args.max_epochs):
                ckpt_path = os.path.join(ckpt_dir, f'dsp_stage_epoch{epoch:02d}.pth')
                torch.save(checkpoint, ckpt_path)
                proto_bank.save(os.path.join(ckpt_dir, f'proto_bank_epoch{epoch:02d}.pth'))
                print(f'  Saved -> {ckpt_path}')
            else:
                print(f'  Skipped periodic checkpoint this epoch (--ckpt-interval={args.ckpt_interval})')
            if is_best:
                torch.save(checkpoint, os.path.join(ckpt_dir, 'best.pth'))
                print(f'  New best val mIoU -> {ckpt_dir}/best.pth')

            panels = save_training_viz(model, val_sample, args.img_dir, device, viz_dir, epoch)
            log_viz_to_tensorboard(epoch_writer, panels, epoch)
            iter_writer.flush()
            epoch_writer.flush()

            if stop_early:
                print(f'\nEarly stopping at epoch {epoch}: val loss did not drop for '
                      f'{epochs_no_improve} consecutive evaluations (patience={args.early_stop_patience}, '
                      f'best val_loss={best_val_loss:.4f}).')
                break

    iter_writer.close()
    epoch_writer.close()
    hook.remove()
    print(f"\nFINAL stage training complete. Best val mIoU vs {args.val_pseudo_label}: "
          f"{best_val_miou:.2f} (initial model: {baseline[miou_key]:.2f}) -> {ckpt_dir}/best.pth")
    if best_val_miou <= baseline[miou_key]:
        print('  No trained epoch beat the initial model on the held-out val set.')
    print('  Evaluate best.pth once on Cityscapes val (test) with evaluate.py --config ... --checkpoint <best.pth>.')


if __name__ == '__main__':
    main()
