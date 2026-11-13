"""Segmentation backbones used by ReFuSe.

Both are wrapped so that ``model(x, out_size)`` returns (1, 19, *out_size)
logits, which is the only calling convention the training scripts use.

  deeplabv2  DeepLabV2 + ResNet-101 (AdaptSegNet / MaxSquareLoss layout).
             Caffe preprocessing (BGR, mean subtraction), input 512x1024.
             Prototype feature = layer4 output (2048-d, stride 8).
  mitb5      MiT-B5 + DAFormer head (mmseg EncoderDecoder from the
             DAFormer/HRDA code base). mmseg test pipeline (RGB, ImageNet
             mean/std), input 512x1024. Prototype feature = the DAFormer
             head's fuse_layer output (256-d, stride 4).
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import NUM_CLASSES, SYNTHIA16

INFER_H, INFER_W = 512, 1024


# ---------------------------------------------------------------------------
# DeepLabV2 + ResNet-101 (exactly the layout of the MaxSquareLoss checkpoints)
# ---------------------------------------------------------------------------

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        # stride in conv1 (1x1), not conv2 -- matches the Caffe-style checkpoints
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ClassifierModule(nn.Module):
    """ASPP head of deeplab_multi.py, including its early return: only the
    first two branches (dilations 6 and 12) contribute."""

    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super().__init__()
        self.conv2d_list = nn.ModuleList([
            nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=p, dilation=d, bias=True)
            for d, p in zip(dilation_series, padding_series)])
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out


class DeepLabV2(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self._make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self._make_layer(256, 23, stride=1, dilation=2)
        self.layer4 = self._make_layer(512, 3, stride=1, dilation=4)
        self.layer5 = ClassifierModule(1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)   # aux, unused
        self.layer6 = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

    def _make_layer(self, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4 or dilation in (2, 4):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4, affine=True))
        layers = [Bottleneck(self.inplanes, planes, stride, dilation, downsample)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x, out_size=None):
        target_size = x.shape[2:] if out_size is None else out_size
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        logits = self.layer6(x)
        return F.interpolate(logits, size=target_size, mode='bilinear', align_corners=True)


def load_deeplabv2_state(path):
    """19-class DeepLabV2 state dict from any supported checkpoint: a flat
    state dict, {'model': ...}, {'state_dict': ...} (MaxSquareLoss, 'module.'
    prefix stripped) or a ReFuSe checkpoint ({'target_model': ...}). A 16-output
    SYNTHIA head is scattered to the 19 train IDs; terrain/truck/train get zero
    weights and a -1e4 bias, so they are never predicted (and get no gradient)."""
    ckpt = torch.load(path, map_location='cpu')
    for key in ('target_model', 'model', 'state_dict'):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
            break
    state = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in ckpt.items()}
    if state['layer6.conv2d_list.0.bias'].shape[0] == len(SYNTHIA16):
        for k, v in list(state.items()):
            if k.startswith(('layer5.', 'layer6.')):
                full = v.new_zeros((NUM_CLASSES,) + tuple(v.shape[1:]))
                if k.endswith('.bias'):
                    full.fill_(-1e4)
                full[SYNTHIA16] = v
                state[k] = full
    return state


# ---------------------------------------------------------------------------
# MiT-B5 + DAFormer head (mmseg)
# ---------------------------------------------------------------------------

class MiTB5Seg(nn.Module):
    """mmseg EncoderDecoder with the DeepLabV2 calling convention. Same two
    resizes as mmseg whole_inference: head output -> input size -> out_size."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, out_size=None):
        from mmseg.ops import resize
        logits = self.net.encode_decode(x, None)
        if out_size is not None:
            logits = resize(logits, size=out_size, mode='bilinear',
                            align_corners=self.net.align_corners, warning=False)
        return logits


def build_mmseg_model(config_path, checkpoint_path, device, absent_classes=()):
    import mmcv
    from mmcv.runner import load_checkpoint
    from mmseg.models import build_segmentor
    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu',
                    revise_keys=[(r'^module\.', ''), ('model.', '')])
    model.cfg = cfg
    if absent_classes:                       # classes the source domain never has
        conv_seg = model.decode_head.conv_seg
        with torch.no_grad():
            conv_seg.weight[list(absent_classes)] = 0.0
            conv_seg.bias[list(absent_classes)] = -1e4
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Deterministic logit upsampling (training path only)
# ---------------------------------------------------------------------------

_upsample_mat_cache = {}


def _bilinear_matrix(n_out, n_in, device, align_corners):
    """(n_out, n_in) float64 1-D bilinear interpolation weights (F.interpolate geometry)."""
    key = (n_out, n_in, str(device), align_corners)
    if key not in _upsample_mat_cache:
        dst = torch.arange(n_out, device=device, dtype=torch.float64)
        if align_corners:
            src = dst * (n_in - 1) / max(1, n_out - 1)
        else:
            src = ((dst + 0.5) * n_in / n_out - 0.5).clamp(min=0)
        i0 = src.floor().long().clamp(max=n_in - 1)
        i1 = (i0 + 1).clamp(max=n_in - 1)
        w1 = src - i0
        mat = torch.zeros(n_out, n_in, device=device, dtype=torch.float64)
        rows = torch.arange(n_out, device=device)
        mat[rows, i0] += 1 - w1
        mat[rows, i1] += w1
        _upsample_mat_cache[key] = mat
    return _upsample_mat_cache[key]


# ---------------------------------------------------------------------------
# Backbone interface
# ---------------------------------------------------------------------------

class Backbone:
    """name, feat_dim, feat_align_corners (geometry of the prototype feature
    map), default_lr, default_pc_cell_offset, and the model-specific ops."""

    def __init__(self, source_ckpt, absent_classes=()):
        self.source_ckpt = source_ckpt
        self.absent_classes = tuple(absent_classes)

    @staticmethod
    def create(name, source_ckpt, source_config=None, absent_classes=(), mmseg_root=None):
        if name == 'deeplabv2':
            return DeepLabV2Backbone(source_ckpt, absent_classes)
        if name == 'mitb5':
            return MiTB5Backbone(source_ckpt, source_config, absent_classes, mmseg_root)
        raise ValueError(f'unknown backbone {name}')

    def predict(self, model, img_bgr, device):
        """(H,W) uint8 class ids at the image's own resolution."""
        with torch.no_grad():
            logits = model(self.preprocess(img_bgr).to(device), out_size=img_bgr.shape[:2])
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


class DeepLabV2Backbone(Backbone):
    name = 'deeplabv2'
    feat_dim = 2048
    feat_align_corners = True
    default_lr = 1e-5
    default_pc_cell_offset = 4                 # layer4 cells are ~16 px apart at 1024x2048
    MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)

    def load_weights(self, path):
        return load_deeplabv2_state(path)

    def build(self, device, weights=None):
        """Source-only model, or the target_model weights of a ReFuSe checkpoint."""
        model = DeepLabV2(NUM_CLASSES)
        model.load_state_dict(self.load_weights(weights or self.source_ckpt), strict=True)
        return model.to(device)

    def preprocess(self, img_bgr):
        resized = cv2.resize(img_bgr, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
        img = resized.astype(np.float32) - self.MEAN_BGR
        return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    def feature_layer(self, model):
        return model.layer4

    def model_logits(self, model, x, out_hw):
        """forward(), but the final upsample is two matmuls (deterministic backward)."""
        x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        x = model.layer4(model.layer3(model.layer2(model.layer1(x))))   # layer4 hook fires
        logits = model.layer6(x)
        wy = _bilinear_matrix(out_hw[0], logits.shape[2], logits.device, True).float()
        wx = _bilinear_matrix(out_hw[1], logits.shape[3], logits.device, True).float()
        return torch.matmul(torch.matmul(wy, logits), wx.t())


class MiTB5Backbone(Backbone):
    name = 'mitb5'
    feat_dim = 256
    feat_align_corners = False
    default_lr = 6e-5
    default_pc_cell_offset = 2                 # fuse_layer cells are ~8 px apart at 1024x2048
    MEAN_RGB = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    STD_RGB = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __init__(self, source_ckpt, source_config, absent_classes=(), mmseg_root=None):
        super().__init__(source_ckpt, absent_classes)
        if not source_config:
            raise ValueError('backbone mitb5 needs source_config (the mmseg config of the source model)')
        self.source_config = source_config
        if mmseg_root and mmseg_root not in sys.path:
            sys.path.insert(0, mmseg_root)     # the DAFormer/HRDA repo (its own mmseg)

    def load_weights(self, path):
        state = torch.load(path, map_location='cpu')
        if isinstance(state, dict) and 'target_model' in state:
            state = state['target_model']
        return state

    def build(self, device, weights=None):
        """Source-only model; or, with weights, a ReFuSe checkpoint ('target_model',
        'net.'-prefixed keys) or another mmseg checkpoint ('state_dict')."""
        mmseg_ckpt, state = self.source_ckpt, None
        if weights is not None:
            raw = torch.load(weights, map_location='cpu')
            if isinstance(raw, dict) and 'target_model' in raw:
                state = raw['target_model']
            elif isinstance(raw, dict) and 'state_dict' in raw:
                mmseg_ckpt = weights
            else:
                state = raw
        model = MiTB5Seg(build_mmseg_model(self.source_config, mmseg_ckpt, device,
                                           self.absent_classes)).to(device)
        if state is not None:
            model.load_state_dict(state, strict=True)
        return model

    def preprocess(self, img_bgr):
        resized = cv2.resize(img_bgr, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
        img = (cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) - self.MEAN_RGB) / self.STD_RGB
        return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    def feature_layer(self, model):
        return model.net.decode_head.fuse_layer

    def model_logits(self, model, x, out_hw):
        """forward(), but its two resizes are composed into one pair of matmuls."""
        net = model.net
        logits = net.decode_head(net.extract_feat(x))                    # fuse_layer hook fires
        ac = net.align_corners
        key = ('composed', tuple(out_hw), tuple(x.shape[2:]), tuple(logits.shape[2:]), str(logits.device), ac)
        if key not in _upsample_mat_cache:
            wy = (_bilinear_matrix(out_hw[0], x.shape[2], logits.device, ac)
                  @ _bilinear_matrix(x.shape[2], logits.shape[2], logits.device, ac))
            wx = (_bilinear_matrix(out_hw[1], x.shape[3], logits.device, ac)
                  @ _bilinear_matrix(x.shape[3], logits.shape[3], logits.device, ac))
            _upsample_mat_cache[key] = (wy.float(), wx.float())
        wy, wx = _upsample_mat_cache[key]
        return torch.matmul(torch.matmul(wy, logits), wx.t())


def freeze_bn(model):
    """Batch size is 1 per image -- keep every BatchNorm layer in eval mode."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
