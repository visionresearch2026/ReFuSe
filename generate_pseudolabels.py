"""
Source pseudo-labels Ps: the source-only model's argmax on every Cityscapes
training image. Stage 1 uses them as the epoch-1 reference and as the anchor
of the VLM-blind class rule; Stage 2 uses them for the held-out consensus
label. Run once per experiment config, before Stage 1.

    python generate_pseudolabels.py --config configs/gta5_deeplabv2_corrclip.yaml --gpu 0

Writes <data.ps_root>/<city>/<stem>_pseudo_label.png (uint8 train IDs; for a
SYNTHIA source, terrain/truck/train never occur). Existing files are kept.
No Cityscapes label is read.
"""
import argparse
import os
import sys

import cv2
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refuse.config import load_config, make_backbone  # noqa: E402

IMG_SUFFIX = '_leftImg8bit.png'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    bb = make_backbone(cfg)
    device = torch.device(f'cuda:{args.gpu}')
    model = bb.build(device).eval()
    img_root, out_root = cfg['data']['train_img_dir'], cfg['data']['ps_root']
    todo = []
    for city in sorted(os.listdir(img_root)):
        if os.path.isdir(os.path.join(img_root, city)):
            todo += [(city, f) for f in sorted(os.listdir(os.path.join(img_root, city))) if f.endswith(IMG_SUFFIX)]
    n_new = 0
    for city, fname in tqdm(todo, desc='pseudo-labels'):
        out = os.path.join(out_root, city, fname.replace(IMG_SUFFIX, '_pseudo_label.png'))
        if os.path.exists(out):
            continue
        img = cv2.imread(os.path.join(img_root, city, fname))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, bb.predict(model, img, device))
        n_new += 1
    print(f'{len(todo)} train images, {n_new} new pseudo-labels -> {out_root}')


if __name__ == '__main__':
    main()
