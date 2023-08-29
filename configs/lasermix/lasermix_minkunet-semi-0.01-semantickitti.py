_base_ = [
    '../_base_/datasets/semi_semantickitti_seg.py',
    '../_base_/default_runtime.py'
]

grid_shape = [480, 360, 32]
segmentor = dict(
    type='MinkUNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='minkunet',
        batch_first=False,
        max_voxels=80000,
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=[-100, -100, -20, 100, 100, 20],
            voxel_size=[0.05, 0.05, 0.05],
            max_voxels=(-1, -1))),
    backbone=dict(
        type='MinkUNetBackbone',
        in_channels=4,
        num_stages=4,
        base_channels=32,
        encoder_channels=[32, 64, 128, 256],
        encoder_blocks=[2, 2, 2, 2],
        decoder_channels=[256, 128, 96, 96],
        decoder_blocks=[2, 2, 2, 2],
        block_type='basic',
        sparseconv_backend='torchsparse'),
    decode_head=dict(
        type='MinkUNetHead',
        channels=96,
        num_classes=19,
        batch_first=False,
        dropout_ratio=0,
        loss_decode=dict(type='mmdet.CrossEntropyLoss', avg_non_ignore=True),
        ignore_index=19),
    train_cfg=dict(),
    test_cfg=dict())

model = dict(
    type='LaserMix',
    segmentor=segmentor,
    data_preprocessor=dict(
        type='MultiBranch3DDataPreprocessor',
        data_preprocessor=dict(
            type='Det3DDataPreprocessor',
            voxel=True,
            voxel_type='minkunet',
            batch_first=False,
            max_voxels=80000,
            voxel_layer=dict(
                max_num_points=-1,
                point_cloud_range=[-100, -100, -20, 100, 100, 20],
                voxel_size=[0.05, 0.05, 0.05],
                max_voxels=(-1, -1)))),
    semi_train_cfg=dict(
        freeze_teacher=True,
        pseudo_thr=0.9,
        ignore_label=19,
        pitch_angles=[-25, 3],
        num_areas=[3, 4, 5, 6]),
    semi_test_cfg=dict(extract_feat_on='teacher', predict_on='teacher'))

lr = 0.24
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='SGD', lr=lr, weight_decay=0.0001, momentum=0.9, nesterov=True))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.008, by_epoch=False, begin=0, end=125),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=15,
        by_epoch=True,
        eta_min=1e-5,
        convert_to_iter_based=True)
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=180000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))
randomness = dict(seed=0, deterministic=False, diff_rank_seed=True)
env_cfg = dict(cudnn_benchmark=True)

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, max_keep_ckpts=2))
log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='mmdet.MeanTeacherHook', momentum=0.01)]
