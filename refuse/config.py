"""Experiment configs (configs/*.yaml): which source domain, backbone and VLM,
and where the data, source model and outputs are."""
import os

import yaml

from .common import SYNTHIA_ABSENT

SOURCES = ('gta5', 'synthia')


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get('source') not in SOURCES:
        raise ValueError(f"{path}: source must be one of {SOURCES}")
    for key in ('backbone', 'source_ckpt', 'data', 'vlm', 'work_dir'):
        if key not in cfg:
            raise ValueError(f'{path}: missing key {key!r}')
    cfg['config_path'] = os.path.abspath(path)
    cfg['absent_classes'] = list(SYNTHIA_ABSENT) if cfg['source'] == 'synthia' else []
    data = cfg['data']
    data.setdefault('train_img_dir', os.path.join(data['cityscapes_root'], 'leftImg8bit', 'train'))
    data.setdefault('val_img_dir', os.path.join(data['cityscapes_root'], 'leftImg8bit', 'val'))
    data.setdefault('val_gt_dir', os.path.join(data['cityscapes_root'], 'gtFine', 'val'))
    return cfg


def make_backbone(cfg):
    from .backbones import Backbone
    return Backbone.create(cfg['backbone'], cfg['source_ckpt'], cfg.get('source_config'),
                           cfg['absent_classes'], cfg.get('mmseg_root'))


def default_text_z_classes(absent_classes):
    """Case 1 labels whose regions define the text z-score statistics (classes
    absent from the source domain are left out)."""
    names = ('building', 'wall', 'fence', 'person', 'rider', 'car', 'truck', 'bus', 'train',
             'motorcycle', 'bicycle')
    from .common import CITYSCAPES_CLASSES
    return ','.join(n for n in names if CITYSCAPES_CLASSES.index(n) not in absent_classes)
