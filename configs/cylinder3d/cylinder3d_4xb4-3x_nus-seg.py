_base_ = [
    '../_base_/datasets/nus-seg.py', '../_base_/models/cylinder3d.py',
    '../_base_/default_runtime.py'
]

grid_shape = [480, 360, 32]
point_cloud_range = [0, -3.14159265359, -5, 50, 3.14159265359, 3]

model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(
            grid_shape=grid_shape, point_cloud_range=point_cloud_range)),
    voxel_encoder=dict(
        grid_shape=grid_shape, point_cloud_range=point_cloud_range),
    backbone=dict(grid_size=grid_shape),
    decode_head=dict(num_classes=17, ignore_index=16))

# optimizer
lr = 0.001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[30],
        gamma=0.1)
]

train_dataloader = dict(batch_size=4)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))
