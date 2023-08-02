# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from .semi_base import SemiBase3DSegmentor


@MODELS.register_module()
class LaserMix(SemiBase3DSegmentor):

    def __init__(self,
                 segmentor: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(LaserMix, self).__init__(
            segmentor=segmentor,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
