"""VLM teachers. Both expose the same interface to the training scripts:

  vlm.text_embed               (19, 512) L2-normalised class text embeddings
  vlm.required_paths(city, stem)  files an image needs ({} for on-the-fly VLMs)
  vlm.maps(entry, device)      None, or a dict with
        label       (H, W) long   VLM label used for Case 1 / agreement
        probs       (19, H, W)    VLM class probabilities (Pc_mean, Case 2 teacher)
        feat        (512, h, w)   L2-normalised dense VLM feature map (CLIP space)
        raw_label   (H, W) long   argmax of probs (Stage 2: consensus / boundary terms)
  vlm.label(entry, device)     (H, W) long label for the held-out metrics, or None
                               (may contain classes listed in absent_classes; the
                               callers ignore those pixels)

  corrclip   Pre-extracted CorrCLIP outputs (SAM2-voted label PNG and the dense
             feature map per image). probs = softmax(40 * cos(feat, text)) at the
             feature resolution, bilinearly upsampled to 1024x2048.
  proxyclip  ProxyCLIP run on the fly on the raw image (no SAM2 voting); probs
             is ProxyCLIP's own full-resolution softmax (logit scale 40).

absent_classes (SYNTHIA: terrain, truck, train) are removed from the VLM's
vocabulary: their logits are -inf before the softmax, and CorrCLIP label
pixels of an absent class are relabelled to the argmax over the other classes.
"""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import NUM_CLASSES, TARGET_H, TARGET_W

CLIP_LOGIT_SCALE = 40.0


class VLM:
    @staticmethod
    def create(cfg, device, absent_classes=()):
        name = cfg['name']
        if name == 'corrclip':
            return CorrCLIPCache(cfg, device, absent_classes)
        if name == 'proxyclip':
            return ProxyCLIPOnline(cfg, device, absent_classes)
        raise ValueError(f'unknown vlm {name}')

    def calib_paths(self):
        """Paths recorded in the threshold-calibration config."""
        return {}


class CorrCLIPCache(VLM):
    name = 'CorrCLIP'
    SEG_SUFFIX = '_leftImg8bit.png'
    FEAT_SUFFIX = '_leftImg8bit.pt'

    def __init__(self, cfg, device, absent_classes=()):
        self.seg_root = cfg['seg_root']
        self.feat_root = cfg['feat_root']
        self.absent = list(absent_classes)
        self.absent_t = torch.tensor(self.absent, dtype=torch.long, device=device)
        text = torch.load(cfg['text_embed'], map_location='cpu')
        self.text_embed = F.normalize(text['embeddings'].float(), dim=-1).to(device)   # (19, 512)

    def required_paths(self, city, stem):
        return {'pc_seg_path': os.path.join(self.seg_root, city, stem + self.SEG_SUFFIX),
                'pc_feat_path': os.path.join(self.feat_root, city, stem + self.FEAT_SUFFIX)}

    def calib_paths(self):
        return {'pc_dir': os.path.abspath(self.seg_root), 'pc_feats_dir': os.path.abspath(self.feat_root)}

    def label(self, entry, device):
        pc = cv2.imread(entry['pc_seg_path'], cv2.IMREAD_GRAYSCALE)
        if pc is None or pc.shape != (TARGET_H, TARGET_W):
            return None
        return torch.from_numpy(pc.astype(np.int64)).to(device)

    def maps(self, entry, device):
        label = self.label(entry, device)
        if label is None:
            return None
        feat = torch.load(entry['pc_feat_path'], map_location='cpu').float().to(device)      # (512,h,w)
        with torch.no_grad():
            feat = F.normalize(feat, dim=0)
            logits = torch.einsum('chw,tc->thw', feat, self.text_embed) * CLIP_LOGIT_SCALE   # (19,h,w)
            if self.absent:
                logits[self.absent] = float('-inf')
            probs = F.interpolate(F.softmax(logits, dim=0).unsqueeze(0), size=(TARGET_H, TARGET_W),
                                  mode='bilinear', align_corners=False)[0]                    # (19,H,W)
            if self.absent:
                absent_px = torch.isin(label, self.absent_t)
                if absent_px.any():
                    label = torch.where(absent_px, probs.argmax(dim=0), label)
            # Raw (pre-SAM2-voting) prediction: the SAM2-voted label is near-constant
            # inside a SAM2 region, so consensus/boundary terms on it would be ~1.
            raw_label = probs.argmax(dim=0)
        return {'label': label, 'probs': probs, 'feat': feat, 'raw_label': raw_label}


class ProxyCLIPOnline(VLM):
    name = 'ProxyCLIP'
    CLIP_INPUT_SIZE = (2048, 448)        # (max W, max H): 1024x2048 -> 448x896

    def __init__(self, cfg, device, absent_classes=()):
        os.environ['PROXYCLIP_ROOT'] = cfg['proxyclip_root']
        from .proxyclip_extractor import (ProxyCLIPFeatureExtractor, load_and_preprocess_image,
                                          CITYSCAPES_CLASS_NAMES_FILE)
        self._load = load_and_preprocess_image
        self.name_path = cfg.get('name_path') or CITYSCAPES_CLASS_NAMES_FILE
        self.extractor = ProxyCLIPFeatureExtractor(
            model_type=cfg.get('clip_path', 'ViT-B/16'), name_path=self.name_path, device=device,
            logit_scale=CLIP_LOGIT_SCALE, prob_thd=0.0,
            slide_stride=cfg.get('slide_stride', 112), slide_crop=cfg.get('slide_crop', 224))
        self.extractor.eval()
        if self.extractor.num_queries != NUM_CLASSES or self.extractor.num_classes != NUM_CLASSES:
            raise ValueError(f'{self.name_path} must list exactly {NUM_CLASSES} classes, one per line')
        self.absent = list(absent_classes)
        self.text_embed = F.normalize(self.extractor.query_features.float(), dim=-1)    # (19, 512)

    def required_paths(self, city, stem):
        return {}

    def calib_paths(self):
        return {'name_path': os.path.abspath(self.name_path)}

    @torch.no_grad()
    def maps(self, entry, device):
        img, (ori_w, ori_h), _ = self._load(entry['img_path'], self.CLIP_INPUT_SIZE)
        if (ori_h, ori_w) != (TARGET_H, TARGET_W):
            return None
        ex = self.extractor
        feat, logits = ex.forward_slide(img.to(device), (TARGET_H, TARGET_W), ex.slide_stride, ex.slide_crop)
        logits = logits.float()
        if self.absent:
            logits[:, self.absent] = float('-inf')
        label, probs = ex._postprocess_logits_batch(logits)
        label = label[0].long()
        return {'label': label, 'probs': probs[0].float(), 'feat': F.normalize(feat[0].float(), dim=0),
                'raw_label': label}

    def label(self, entry, device):
        m = self.maps(entry, device)
        return None if m is None else m['label']
