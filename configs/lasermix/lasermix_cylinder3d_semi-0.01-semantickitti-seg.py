_base_ = [
    '../_base_/datasets/semi_semantickitti_seg.py',
    '../_base_/schedules/schedule-3x.py', '../_base_/default_runtime.py'
]

grid_shape = [480, 360, 32]
segmentor = dict(
    type='Cylinder3D',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=grid_shape,
            point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 2],
            max_num_points=-1,
            max_voxels=-1)),
    voxel_encoder=dict(
        type='SegVFE',
        feat_channels=[64, 128, 256, 256],
        in_channels=6,
        with_voxel_center=True,
        feat_compression=16),
    backbone=dict(
        type='Asymm3DSpconv',
        grid_size=grid_shape,
        input_channels=16,
        base_channels=32,
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1)),
    decode_head=dict(
        type='Cylinder3DHead',
        channels=128,
        num_classes=20,
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0),
        loss_lovasz=dict(type='LovaszLoss', loss_weight=1.0,
                         reduction='none')),
    train_cfg=None,
    test_cfg=dict(mode='whole'))

model = dict(
    type='LaserMix',
    segmentor=segmentor,
    data_preprocessor=dict(
        type='MultiBranch3DDataPreprocessor',
        data_preprocessor=dict(
            type='Det3DDataPreprocessor',
            voxel=True,
            voxel_type='cylindrical',
            voxel_layer=dict(
                grid_shape=grid_shape,
                point_cloud_range=[
                    0, -3.14159265359, -4, 50, 3.14159265359, 2
                ],
                max_num_points=-1,
                max_voxels=-1))),
    semi_train_cfg=dict(freeze_teacher=True, pseudo_thr=0.9, ignore_label=19),
    semi_test_cfg=dict(extract_feat_on='student', predict_on='student'))

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=180000,
    val_interval=5000)

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, max_keep_ckpts=2))
log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='mmdet.MeanTeacherHook', momentum=0.01)]
