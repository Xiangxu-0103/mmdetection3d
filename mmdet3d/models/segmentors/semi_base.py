# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from torch import Tensor, nn

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import Base3DSegmentor


@MODELS.register_module()
class SemiBase3DSegmentor(Base3DSegmentor):
    """Base class for semi-supervisied segmentors.

    Semi-supervisied segmentors typically consisting of a teacher model updated
    by exponential moving average and a student model updated by gradient
    descent.

    Args:
        segmentor (:obj:`ConfigDict` or dict): The segmentor config.
        semi_train_cfg (:obj:`ConfigDict` or dict, optional): The
            semi-supervised training config. Defaults to None.
        semi_test_cfg (:obj:`ConfigDict` or dict, optional): The semi-segmentor
            testing config. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`Det3DDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or List[:obj:`ConfigDict` or dict],
            optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 segmentor: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(SemiBase3DSegmentor, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.student = MODELS.build(segmentor)
        self.teacher = MODELS.build(segmentor)
        self.semi_train_cfg = semi_train_cfg
        self.semi_test_cfg = semi_test_cfg
        if self.semi_train_cfg.get('freeze_teacher', True) is True:
            self.freeze(self.teacher)

    @staticmethod
    def freeze(model: nn.Module) -> None:
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def loss(self, multi_batch_inputs: Dict[str, dict],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    def predict(self, batch_inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        pass

    def _forward(self,
                 batch_inputs: dict,
                 batch_data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def extract_feat(self, batch_inputs: Tensor) -> dict:
        if self.semi_test_cfg.get('extract_feat_on', 'teacher') == 'teacher':
            return self.teacher.extract_feat(batch_inputs)
        else:
            return self.student.extract_feat(batch_inputs)

    def encode_decode(self, batch_inputs: Tensor,
                      batch_data_samples: SampleList) -> Tensor:
        """Placeholder for encode images with backbone and decode into a
        semantic segmentation map of the same size as input."""
        pass
