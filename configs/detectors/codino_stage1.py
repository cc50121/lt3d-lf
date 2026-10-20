import os

_base_ = '../../third_party/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py'

class_name = (
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'emergency_vehicle', 'adult', 'child', 'police_officer',
    'construction_worker', 'stroller', 'personal_mobility',
    'pushable_pullable', 'debris', 'traffic_cone', 'barrier')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

data_root = os.environ.get('LT3D_MERGED_ROOT', 'data/nuscenes_nuimages/')
data_val_root = os.environ.get('LT3D_DATA_ROOT', 'data/nuscenes/').rstrip('/') + '/'

model = _base_.model
model.query_head.num_classes = num_classes
model.roi_head[0].bbox_head.num_classes = num_classes
model.bbox_head[0].num_classes = num_classes
# Avoid a separate GitHub Swin-L backbone download; load full Co-DINO O365 checkpoint via load_from.
model.backbone.init_cfg = None

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_2D_nuscenes_nuimages.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_val_root,
        metainfo=metainfo,
        ann_file='test_2D_val.json',
        data_prefix=dict(img=os.environ.get('LT3D_VAL_IMAGE_PREFIX', '')),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_val_root + 'test_2D_val.json',
    metric='bbox',
    classwise=True,
    outfile_prefix='outputs/ijcv_revision/codino_swinl_o365_nuscenes_nuimages')
test_evaluator = val_evaluator

max_epochs = 6
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[5],
        gamma=0.1)
]

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=6,
        save_best='coco/bbox_mAP',
        rule='greater'),
    logger=dict(type='LoggerHook', interval=100))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl', timeout=21600))

log_processor = dict(by_epoch=True)
