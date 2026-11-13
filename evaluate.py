"""
Final evaluation on the 500 Cityscapes val images (run once per trained model;
val is never used for training or model selection).

    python evaluate.py --config configs/gta5_deeplabv2_corrclip.yaml \
        --checkpoint <work_dir>/stage2/best.pth
    python evaluate.py --config ... --checkpoint <source checkpoint>   # source-only baseline
    python evaluate.py --config ... --checkpoint ... --save-preds       # also save every prediction

--checkpoint accepts a ReFuSe checkpoint (Stage 1 / Stage 2, 'target_model')
or the source-only weights of the config's backbone.

Metrics
  gta5     19-class mIoU.
  synthia  mIoU16 and mIoU13 (without wall, fence, pole) under two protocols,
           both from one forward pass:
             16-class head  GT terrain/truck/train -> ignore and predictions
                            restricted to the 16 SYNTHIA classes (DeepLabV2
                            SFUDA convention; used for the deeplabv2 tables)
             DAFormer       ordinary 19-class evaluation, mean over the 16 / 13
                            classes (used for the mitb5 tables)

Outputs (--out-dir, default <checkpoint dir>/eval_results):
  results_<checkpoint name>.json, image | GT | prediction panels for
  --viz-count images, and with --save-preds:
  pred_labelTrainIds/<city>/<stem>_pred.png, pred_colored/<city>/<stem>_pred_colored.png
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refuse.common import (NUM_CLASSES, IGNORE_LABEL, CITYSCAPES_CLASSES, SYNTHIA_ABSENT,  # noqa: E402
                           SYNTHIA16, SYNTHIA13, colorize)
from refuse.config import load_config, make_backbone  # noqa: E402

IMG_SUFFIX = '_leftImg8bit.png'


def update_confusion(matrix, pred, gt):
    mask = gt != IGNORE_LABEL
    idx = gt[mask].astype(np.int64) * NUM_CLASSES + pred[mask].astype(np.int64)
    matrix += np.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def per_class_iou(matrix):
    inter = np.diag(matrix).astype(np.float64)
    union = matrix.sum(1) + matrix.sum(0) - np.diag(matrix)
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


def summarize(matrix):
    iou = per_class_iou(matrix)
    return {'miou19': float(np.nanmean(iou)),
            'miou16': float(np.nanmean(iou[SYNTHIA16])),
            'miou13': float(np.nanmean(iou[SYNTHIA13])),
            'per_class_iou': {CITYSCAPES_CLASSES[c]: (None if np.isnan(iou[c]) else float(iou[c]))
                              for c in range(NUM_CLASSES)}}


def collect_val_images(val_img_dir):
    paths = []
    for city in sorted(os.listdir(val_img_dir)):
        city_dir = os.path.join(val_img_dir, city)
        if os.path.isdir(city_dir):
            paths += [(city, f) for f in sorted(os.listdir(city_dir)) if f.endswith(IMG_SUFFIX)]
    return paths


def make_panel(img_bgr, gt, pred, scale=0.5):
    h, w = img_bgr.shape[:2]
    size = (int(w * scale), int(h * scale))
    panels = [cv2.resize(p, size, interpolation=cv2.INTER_NEAREST)
              for p in (img_bgr, colorize(gt) if gt is not None else np.zeros_like(img_bgr), colorize(pred))]
    for panel, text in zip(panels, ('image', 'ground truth', 'prediction')):
        cv2.putText(panel, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return np.hstack(panels)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--viz-count', type=int, default=12,
                        help='image | GT | prediction panels for this many evenly spaced val images')
    parser.add_argument('--save-preds', action='store_true',
                        help='save the prediction of every val image (train-ID PNG + coloured PNG)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    synthia = cfg['source'] == 'synthia'
    bb = make_backbone(cfg)
    device = torch.device(f'cuda:{args.gpu}')
    ckpt_path = os.path.abspath(args.checkpoint)
    model = bb.build(device, ckpt_path).eval()
    out_dir = args.out_dir or os.path.join(os.path.dirname(ckpt_path), 'eval_results')
    os.makedirs(out_dir, exist_ok=True)
    print(f'Evaluating {ckpt_path} ({bb.name}, source {cfg["source"]}) -> {out_dir}')

    data = cfg['data']
    val = collect_val_images(data['val_img_dir'])
    viz = set(np.linspace(0, len(val) - 1, num=min(args.viz_count, len(val)), dtype=int).tolist()) \
        if args.viz_count > 0 else set()
    absent = list(SYNTHIA_ABSENT)
    absent_lut = np.arange(256, dtype=np.uint8)
    absent_lut[absent] = IGNORE_LABEL

    conf19 = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)    # 19-class / DAFormer protocol
    conf16 = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)    # SYNTHIA 16-class-head protocol
    for idx, (city, fname) in enumerate(tqdm(val, desc='Cityscapes val')):
        img = cv2.imread(os.path.join(data['val_img_dir'], city, fname))
        gt = cv2.imread(os.path.join(data['val_gt_dir'], city,
                                     fname.replace(IMG_SUFFIX, '_gtFine_labelTrainIds.png')), cv2.IMREAD_GRAYSCALE)
        with torch.no_grad():
            logits = model(bb.preprocess(img).to(device), out_size=img.shape[:2])[0]
        pred = logits.argmax(0).cpu().numpy().astype(np.uint8)
        if synthia:
            logits[absent] = float('-inf')
            pred16 = logits.argmax(0).cpu().numpy().astype(np.uint8)
        else:
            pred16 = pred
        if gt is not None:
            update_confusion(conf19, pred, gt)
            if synthia:
                update_confusion(conf16, pred16, absent_lut[gt])
        shown = pred16 if synthia else pred
        stem = fname[:-len(IMG_SUFFIX)]
        if idx in viz:
            cv2.imwrite(os.path.join(out_dir, f'{city}_{stem}.png'),
                        make_panel(img, None if gt is None else (absent_lut[gt] if synthia else gt), shown))
        if args.save_preds:
            for sub, arr, suffix in (('pred_labelTrainIds', shown, '_pred.png'),
                                     ('pred_colored', colorize(shown), '_pred_colored.png')):
                os.makedirs(os.path.join(out_dir, sub, city), exist_ok=True)
                cv2.imwrite(os.path.join(out_dir, sub, city, stem + suffix), arr)

    r19 = summarize(conf19)
    result = {'checkpoint': ckpt_path, 'config': cfg['config_path'], 'source': cfg['source'],
              'backbone': bb.name, 'n_images': len(val)}
    print('\n' + '=' * 60)
    if synthia:
        r16 = summarize(conf16)
        result.update({'16class_head_protocol': r16, 'daformer_protocol': r19})
        print(f'16-class-head protocol  mIoU16 {r16["miou16"] * 100:6.2f}  mIoU13 {r16["miou13"] * 100:6.2f}')
        print(f'DAFormer protocol       mIoU16 {r19["miou16"] * 100:6.2f}  mIoU13 {r19["miou13"] * 100:6.2f}')
        rows = [(n, r16['per_class_iou'][n], r19['per_class_iou'][n]) for n in CITYSCAPES_CLASSES]
        print(f'{"class":<16}{"16-head":>9}{"DAFormer":>10}')
        for c, (n, a, b) in enumerate(rows):
            if c in absent:
                print(f'{n:<16}{"-":>9}{"-":>10}   (not in SYNTHIA)')
            else:
                print(f'{n:<16}{(a or 0) * 100:9.2f}{(b or 0) * 100:10.2f}')
    else:
        result.update({'miou': r19['miou19'], 'per_class_iou': r19['per_class_iou']})
        print(f'mIoU (19 classes) {r19["miou19"] * 100:6.2f}')
        for n, v in r19['per_class_iou'].items():
            print(f'{n:<16}{"nan" if v is None else f"{v * 100:6.2f}"}')
    print('=' * 60)
    out = os.path.join(out_dir, f'results_{os.path.splitext(os.path.basename(ckpt_path))[0]}.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
