# SYNTHIA (RAND_CITYSCAPES) source-only model, MiT-B5 + DAFormer head (mmseg
# config for the DAFormer/HRDA code base). Fixed split 9000 train / 400 val
# (synthia_split/, seed 0); checkpoint selected on SYNTHIA val. Labels: 16
# SYNTHIA classes as Cityscapes train IDs (convert_synthia_labels16.py);
# terrain/truck/train are ignore, the head keeps 19 outputs.
# Set synthia_root, synthia_split_dir (absolute) and the Cityscapes data_root.
#   python convert_synthia_labels16.py --root <RAND_CITYSCAPES>     (once)
#   PYTHONPATH=<DAFormer/HRDA repo> python tools/train.py <this file>
# ReFuSe itself only uses the model definition and test pipeline of this file.
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(type='mit_b5', style='pytorch'),
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=dict(type='BN', requires_grad=True))),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(work_dir='work_dirs/synthia_source_only_mitb5'),
    test_cfg=dict(mode='whole'))
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()

synthia_root = 'data/Synthia/RAND_CITYSCAPES/'
# absolute: mmseg joins a relative split path onto data_root
synthia_split_dir = '/abs/path/to/ReFuSe/source_models/synthia_split/'   # must be absolute
cityscapes_classes = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                      'traffic light', 'traffic sign', 'vegetation', 'terrain',
                      'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                      'motorcycle', 'bicycle')
cityscapes_palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70],
                      [102, 102, 156], [190, 153, 153], [153, 153, 153],
                      [250, 170, 30], [220, 220, 0], [107, 142, 35],
                      [152, 251, 152], [70, 130, 180], [220, 20, 60],
                      [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                      [0, 80, 100], [0, 0, 230], [119, 11, 32]]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 760)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
# SYNTHIA val is evaluated at the training scale (1280x760)
synthia_val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 760),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='CustomDataset',
        data_root=synthia_root,
        img_dir='RGB',
        img_suffix='.png',
        ann_dir='GT/LABELS16_trainIds',
        seg_map_suffix='_labelTrainIds.png',
        split=synthia_split_dir + 'train.txt',
        classes=cityscapes_classes,
        palette=cityscapes_palette,
        pipeline=train_pipeline),
    val=dict(
        type='CustomDataset',
        data_root=synthia_root,
        img_dir='RGB',
        img_suffix='.png',
        ann_dir='GT/LABELS16_trainIds',
        seg_map_suffix='_labelTrainIds.png',
        split=synthia_split_dir + 'val.txt',
        classes=cityscapes_classes,
        palette=cityscapes_palette,
        pipeline=synthia_val_pipeline),
    test=dict(
        type='CityscapesDataset',
        data_root='data/Cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=2)
evaluation = dict(interval=4000, metric='mIoU', save_best='mIoU')
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
work_dir = 'work_dirs/synthia_source_only_mitb5'
gpu_ids = range(0, 1)
