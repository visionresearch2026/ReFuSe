"""
Convert SYNTHIA RAND_CITYSCAPES GT/LABELS (uint16; class id in the first
channel of the README = cv2 channel 2) to single-channel Cityscapes train-ID
PNGs (SYNTHIA-16 mapping: terrain, truck, train, void, parking-slot,
road-work and lane-marking -> 255). Needed once before training the MiT-B5
SYNTHIA source model with synthia_mitb5_source_only.py.

    python source_models/convert_synthia_labels16.py --root data/Synthia/RAND_CITYSCAPES

Writes <root>/GT/LABELS16_trainIds/<name>_labelTrainIds.png (existing files are kept).
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

IGNORE_LABEL = 255
SYNTHIA_TO_TRAINID = {
    3: 0,    # road
    4: 1,    # sidewalk
    2: 2,    # building
    21: 3,   # wall
    5: 4,    # fence
    7: 5,    # pole
    15: 6,   # traffic light
    9: 7,    # traffic sign
    6: 8,    # vegetation
    1: 10,   # sky
    10: 11,  # person
    17: 12,  # rider
    8: 13,   # car
    19: 15,  # bus
    12: 17,  # motorcycle
    11: 18,  # bicycle
}
LUT = np.full(65536, IGNORE_LABEL, dtype=np.uint8)
for k, v in SYNTHIA_TO_TRAINID.items():
    LUT[k] = v


def convert(name, root, out_dir):
    out = os.path.join(out_dir, name + '_labelTrainIds.png')
    if not os.path.exists(out):
        raw = cv2.imread(os.path.join(root, 'GT', 'LABELS', name + '.png'), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError(f'cannot read label {name}')
        cv2.imwrite(out, LUT[raw[:, :, 2]])
    return name


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='SYNTHIA RAND_CITYSCAPES directory')
    p.add_argument('--nproc', type=int, default=8)
    args = p.parse_args()
    out_dir = os.path.join(args.root, 'GT', 'LABELS16_trainIds')
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n[:-4] for n in os.listdir(os.path.join(args.root, 'GT', 'LABELS')) if n.endswith('.png'))
    with Pool(args.nproc) as pool:
        for _ in tqdm(pool.imap_unordered(partial(convert, root=args.root, out_dir=out_dir), names,
                                          chunksize=32), total=len(names)):
            pass
    print(f'{len(names)} labels -> {out_dir}')


if __name__ == '__main__':
    main()
