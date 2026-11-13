"""Constants shared by every ReFuSe script (Cityscapes label space)."""
import numpy as np

NUM_CLASSES = 19
IGNORE_LABEL = 255
TARGET_H, TARGET_W = 1024, 2048          # Cityscapes label / SAM2 mask / VLM label resolution

CITYSCAPES_CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
                      'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                      'truck', 'bus', 'train', 'motorcycle', 'bicycle')

# SYNTHIA has no terrain (9), truck (14) or train (16). A 16-output SYNTHIA
# head predicts the other 16 train IDs in this order (MaxSquareLoss order).
SYNTHIA_ABSENT = (9, 14, 16)
SYNTHIA16 = [c for c in range(NUM_CLASSES) if c not in SYNTHIA_ABSENT]
SYNTHIA13 = [c for c in SYNTHIA16 if c not in (3, 4, 5)]      # minus wall, fence, pole

# Cityscapes train-ID palette, BGR (for cv2).
CITYSCAPES_PALETTE_BGR = np.array([
    [128, 64, 128], [232, 35, 244], [70, 70, 70], [156, 102, 102], [153, 153, 190],
    [153, 153, 153], [30, 170, 250], [0, 220, 220], [35, 142, 107], [152, 251, 152],
    [180, 130, 70], [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0],
    [100, 60, 0], [100, 80, 0], [230, 0, 0], [32, 11, 119]], dtype=np.uint8)


def colorize(label_map):
    """(H,W) train IDs -> (H,W,3) BGR image; ignore (255) pixels are black."""
    safe = np.where(label_map == IGNORE_LABEL, 0, label_map).astype(np.uint8)
    out = CITYSCAPES_PALETTE_BGR[safe]
    out[label_map == IGNORE_LABEL] = 0
    return out
