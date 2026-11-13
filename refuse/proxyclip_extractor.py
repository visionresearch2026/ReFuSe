"""
Frozen ProxyCLIP (CLIP ViT-B/16 + DINO ViT-B/8 proxy attention) used as the
VLM teacher, run on the fly: returns the dense (512, h, w) patch feature map
(in the CLIP text-embedding space) and per-class logits for one image.

Same math as ProxyCLIP's proxyclip_segmentor.ProxyCLIPSegmentation, with the
intermediate dense features exposed. Needs the official ProxyCLIP repository
(open_clip, myutils, prompts): set the environment variable PROXYCLIP_ROOT to
its path (refuse/vlm.py does this from the config's vlm.proxyclip_root).
"""
import os
import sys

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


_PROXYCLIP_ROOT = os.environ.get('PROXYCLIP_ROOT', '')
if not _PROXYCLIP_ROOT or not os.path.isdir(_PROXYCLIP_ROOT):
    raise ImportError('Set PROXYCLIP_ROOT (or vlm.proxyclip_root in the config) to the ProxyCLIP repository.')
if _PROXYCLIP_ROOT not in sys.path:
    sys.path.insert(0, _PROXYCLIP_ROOT)

from open_clip import create_model, tokenizer
from myutils import UnNormalize
from prompts.imagenet_template import openai_imagenet_template

CITYSCAPES_CLASS_NAMES_FILE = os.path.join(_PROXYCLIP_ROOT, 'configs', 'cls_city_scapes.txt')


def get_cls_idx(path):
    """Same parsing rule as proxyclip_segmentor.py's get_cls_idx: one line per class, semicolon-separated synonyms."""
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split('; ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices


class ProxyCLIPFeatureExtractor(nn.Module):
    """Frozen CLIP+DINO wrapper that returns dense patch features + per-class logits.

    Mirrors the math of `ProxyCLIPSegmentation` (proxyclip_segmentor.py) - same
    open_clip backbone, same DINO-driven proxy attention on the last block
    (`beta`/`gamma`) - but exposes the intermediate dense feature map (before
    the text-query projection) so it can be used to train a separate decoder,
    and adds a batched postprocessing path (`extract_batch_with_confidence`)
    for on-the-fly training instead of one-image-at-a-time disk caching.
    """

    def __init__(self, clip_type='openai', model_type='ViT-B/16', vfm_model='dino',
                 name_path='./configs/cls_city_scapes.txt',
                 device=torch.device('cuda'), logit_scale=40, prob_thd=0.0,
                 beta=1.2, gamma=3.0, slide_stride=112, slide_crop=224):
        super().__init__()
        self.device = device

        self.clip = create_model(model_type, pretrained=clip_type, precision='fp16')
        self.clip.eval().to(device)
        self.tokenizer = tokenizer.tokenize

        if vfm_model != 'dino':
            raise NotImplementedError(
                f"vfm_model={vfm_model!r} not supported (only 'dino', matching this repo's "
                f"configs/base_config.py, is implemented here)")
        self.vfm_model = vfm_model
        self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        self.vfm = self.vfm.half()
        for p in self.vfm.parameters():
            p.requires_grad = False
        self.vfm.eval().to(device)
        self.vfm_patch_size = self.vfm.patch_embed.patch_size

        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        query_words, query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(query_idx) + 1
        self.query_idx = torch.tensor(query_idx, dtype=torch.long, device=device)

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = self.clip.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)

        self.feat_dim = self.clip.visual.output_dim
        self.beta = beta
        self.gamma = gamma
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop

    def forward_feature(self, img):
        """Single forward pass (no sliding window) through the real DINO-proxied CLIP fusion.

        Args:
            img: (B, 3, h, w) CLIP-normalized crop

        Returns:
            feat_map: (B, feat_dim, I, J) L2-normalized dense patch features, I=h//8, J=w//8
                      (DINO ViT-B/8's patch grid - see module docstring)
            logits:   (B, num_queries, h, w) per-query cosine similarity, upsampled to the
                      crop's own pixel resolution
        """
        imgs_norm = torch.stack([self.norm(self.unnorm(img[i])) for i in range(img.shape[0])], dim=0)
        imgs_norm = imgs_norm.half()

        patch = self.vfm_patch_size
        I = imgs_norm.shape[-2] // patch
        J = imgs_norm.shape[-1] // patch
        feat = self.vfm.get_intermediate_layers(imgs_norm)[0]
        ex_feats = feat[:, 1:, :].reshape(feat.shape[0], I, J, -1).permute(0, 3, 1, 2)

        image_features = self.clip.encode_image(img.half(), external_feats=ex_feats, beta=self.beta, gamma=self.gamma)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        feat_map = image_features.permute(0, 2, 1).reshape(-1, self.feat_dim, I, J)

        logits = image_features @ self.query_features.T
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], I, J)
        logits = F.interpolate(logits, size=img.shape[-2:], mode='bilinear', align_corners=False)

        return feat_map, logits

    def compute_padsize(self, H, W, patch_size):
        """Same as ProxyCLIPSegmentation.compute_padsize: symmetric zero-pad up to a multiple of patch_size."""
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l
        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t
        return l, r, t, b

    def forward_slide(self, img, ori_shape, stride=112, crop_size=224):
        """Sliding-window inference, aggregating both the dense feature map and the logits.

        Feature map is aggregated at the DINO-patch grid resolution (H_img/8, W_img/8);
        logits are aggregated at pixel resolution (same as ProxyCLIPSegmentation.forward_slide
        in proxyclip_segmentor.py) and finally resized to `ori_shape`.
        """
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        patch = self.vfm_patch_size

        h_feat, w_feat = h_img // patch, w_img // patch

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        logit_preds = img.new_zeros((batch_size, self.num_queries, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        feat_preds = img.new_zeros((batch_size, self.feat_dim, h_feat, w_feat))
        feat_count = img.new_zeros((batch_size, 1, h_feat, w_feat))

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = img[:, :, y1:y2, x1:x2]

                # Pad to a multiple of 7*patch (56 for the DINO patch=8), same as
                # ProxyCLIPSegmentation.forward_slide, so padding stays grid-aligned with
                # the DINO branch that determines the output token grid.
                H, W = crop_img.shape[2:]
                pad = self.compute_padsize(H, W, 7 * patch)
                if any(pad):
                    crop_img = F.pad(crop_img, pad)

                crop_feat, crop_logit = self.forward_feature(crop_img)

                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_logit = crop_logit[:, :, t:t + H, l:l + W]
                    lg, tg = l // patch, t // patch
                    crop_feat = crop_feat[:, :, tg:tg + H // patch, lg:lg + W // patch]

                logit_preds += F.pad(
                    crop_logit,
                    (int(x1), int(logit_preds.shape[3] - x2), int(y1), int(logit_preds.shape[2] - y2))
                )
                count_mat[:, :, y1:y2, x1:x2] += 1

                fy1, fx1 = y1 // patch, x1 // patch
                fh, fw = crop_feat.shape[-2:]
                fy2, fx2 = min(fy1 + fh, h_feat), min(fx1 + fw, w_feat)
                feat_preds[:, :, fy1:fy2, fx1:fx2] += crop_feat[:, :, :fy2 - fy1, :fx2 - fx1]
                feat_count[:, :, fy1:fy2, fx1:fx2] += 1

        assert (count_mat == 0).sum() == 0
        assert (feat_count == 0).sum() == 0

        logit_preds = logit_preds / count_mat
        feat_preds = feat_preds / feat_count

        logits = F.interpolate(logit_preds, size=ori_shape, mode='bilinear', align_corners=False)

        return feat_preds, logits

    def _postprocess_logits(self, seg_logit):
        """seg_logit: (num_queries, H, W) -> seg_pred (H, W), confidence (num_classes, H, W).
        Same as `ProxyCLIPSegmentation.postprocess_result` (proxyclip_segmentor.py).
        """
        seg_logit = seg_logit * self.logit_scale
        seg_logit = seg_logit.softmax(0)

        if self.num_classes != self.num_queries:
            seg_logit = seg_logit.unsqueeze(0)
            cls_index = F.one_hot(self.query_idx, self.num_classes)
            cls_index = cls_index.T.view(self.num_classes, self.num_queries, 1, 1)
            seg_logit = (seg_logit * cls_index).max(1)[0]

        confidence = seg_logit  # (num_classes, H, W) softmax probabilities per class
        seg_pred = seg_logit.argmax(0)
        seg_pred[seg_logit.max(0)[0] < self.prob_thd] = 0
        return seg_pred, confidence

    def _postprocess_logits_batch(self, seg_logit):
        """Batched version of `_postprocess_logits`.

        seg_logit: (B, num_queries, H, W) -> seg_pred (B, H, W), confidence (B, num_classes, H, W)
        """
        seg_logit = seg_logit * self.logit_scale
        seg_logit = seg_logit.softmax(1)

        if self.num_classes != self.num_queries:
            cls_index = F.one_hot(self.query_idx, self.num_classes).T  # (num_classes, num_queries)
            cls_index = cls_index.view(1, self.num_classes, self.num_queries, 1, 1)
            seg_logit = (seg_logit.unsqueeze(1) * cls_index).max(2)[0]  # (B, num_classes, H, W)

        confidence = seg_logit
        seg_pred = seg_logit.argmax(1)
        seg_pred[seg_logit.max(1)[0] < self.prob_thd] = 0
        return seg_pred, confidence

    @torch.no_grad()
    def extract_batch_with_confidence(self, inputs, ori_shape):
        """Batched, on-the-fly extraction for a batch of preprocessed images that
        share the same original resolution (true for Cityscapes: every frame is
        1024x2048), so training can pull feature/label/confidence maps straight
        from the frozen CLIP+DINO backbone with no intermediate disk I/O.

        Args:
            inputs: (B, 3, H, W) preprocessed CLIP tensor
            ori_shape: (H_orig, W_orig) shared by every image in the batch

        Returns:
            feat_map:       (B, feat_dim, h, w) float32
            seg_pred:       (B, H_orig, W_orig) class-index pseudo-labels
            max_confidence: (B, H_orig, W_orig) float32 softmax prob of the predicted class
        """
        feat_preds, logits = self.forward_slide(inputs, ori_shape, self.slide_stride, self.slide_crop)
        seg_pred, confidence = self._postprocess_logits_batch(logits)
        max_confidence = confidence.max(1)[0]
        return feat_preds.float(), seg_pred, max_confidence

    @torch.no_grad()
    def extract(self, inputs, img_path=None):
        """
        Args:
            inputs: preprocessed image tensor (1, 3, H, W)
            img_path: path to the original image, used only to read the original resolution
                      (kept as a named arg for interface parity with the CorrCLIP extractor)

        Returns:
            feat_map: (feat_dim, H/8, W/8) float16
            seg_pred: (H, W) class-index segmentation map at original resolution
        """
        with Image.open(img_path) as im:
            ori_shape = (im.height, im.width)

        feat_preds, logits = self.forward_slide(inputs, ori_shape, self.slide_stride, self.slide_crop)
        seg_pred, _ = self._postprocess_logits(logits[0])

        return feat_preds[0].half(), seg_pred

    @torch.no_grad()
    def extract_with_confidence(self, inputs, img_path=None):
        """Same as `extract`, but also returns per-class softmax confidence maps."""
        with Image.open(img_path) as im:
            ori_shape = (im.height, im.width)

        feat_preds, logits = self.forward_slide(inputs, ori_shape, self.slide_stride, self.slide_crop)
        seg_pred, confidence = self._postprocess_logits(logits[0])

        return feat_preds[0].half(), seg_pred, confidence


def load_and_preprocess_image(img_path, target_size=(2048, 448)):
    """Load and CLIP-normalize an image, resizing (keeping aspect ratio) to fit target_size."""
    img = Image.open(img_path).convert('RGB')
    orig_size = img.size  # (W, H)

    w, h = orig_size
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    img_tensor = transform(img_resized).unsqueeze(0)

    return img_tensor, orig_size, (new_h, new_w)
