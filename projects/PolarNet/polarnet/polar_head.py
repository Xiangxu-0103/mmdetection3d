# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmdet3d.models import Base3DDecodeHead
from mmdet3d.models.data_preprocessors.voxelize import dynamic_scatter_3d
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType


@MODELS.register_module()
class PolarHead(Base3DDecodeHead):
    """RangeImage decoder head.

    Args:
        loss_lovasz (dict or :obj:`ConfigDict`, optional): Config of Lovasz
            loss. Defaults to None.
    """

    def __init__(self,
                 channels: int,
                 num_classes: int,
                 height: int,
                 loss_lovasz: OptConfigType = None,
                 conv_seg_kernel_size: int = 1,
                 **kwargs) -> None:
        super(PolarHead, self).__init__(
            channels=channels,
            num_classes=num_classes,
            conv_seg_kernel_size=conv_seg_kernel_size,
            **kwargs)
        self.height = height

        self.conv_seg = self.build_conv_seg(
            channels=channels,
            num_classes=num_classes * height,
            kernel_size=conv_seg_kernel_size)
        if loss_lovasz is not None:
            self.loss_lovasz = MODELS.build(loss_lovasz)
        else:
            self.loss_lovasz = None

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> nn.Module:
        return nn.Conv2d(
            channels,
            num_classes,
            kernel_size=kernel_size,
            padding=kernel_size // 2)

    def forward(self, voxel_dict: dict) -> dict:
        """Forward function."""
        logits = self.cls_seg(voxel_dict['voxel_feats'])
        batch_size, _, H, W = logits.size()
        logits = logits.permute(0, 2, 3, 1).contiguous()
        logits = logits.view(batch_size, H, W, self.height, self.num_classes)
        voxel_dict['logits'] = logits
        return voxel_dict

    def loss_by_feat(self, voxel_dict: dict,
                     batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Compute semantic segmentation loss.

        Args:
            voxel_dict (dict): The dict may contain `logits`,
                `point2voxel_map`.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        seg_logit_feat = voxel_dict['logits']

        voxel_semantic_segs = []
        seg_logits = []
        coors = voxel_dict['coors']
        for batch_idx, data_sample in enumerate(batch_data_samples):
            pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
            batch_mask = coors[:, 0] == batch_idx
            this_coors = coors[batch_mask, 1:]
            voxel_semantic_mask, voxel_coor, _ = dynamic_scatter_3d(
                F.one_hot(pts_semantic_mask.long()).float(), this_coors,
                'mean')
            voxel_coor = voxel_coor.long()
            voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)
            voxel_semantic_segs.append(voxel_semantic_mask)
            seg_logits.append(seg_logit_feat[batch_idx, voxel_coor[:, 0],
                                             voxel_coor[:, 1], voxel_coor[:,
                                                                          2]])
        seg_label = torch.cat(voxel_semantic_segs)
        seg_logits = torch.cat(seg_logits)
        loss = dict()
        loss['loss_ce'] = self.loss_ce(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        if self.loss_lovasz:
            loss['loss_lovasz'] = self.loss_lovasz(
                seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

    def predict(self, voxel_dict: dict, batch_input_metas: List[dict],
                test_cfg: ConfigType) -> List[Tensor]:
        """Forward function for testing.

        Args:
            voxel_dict (dict): Features from backbone.
            batch_input_metas (List[dict]): Meta information of a batch of
                samples.
            test_cfg (dict or :obj:`ConfigDict`): The testing config.

        Returns:
            List[Tensor]: List of point-wise segmentation logits.
        """
        voxel_dict = self.forward(voxel_dict)
        seg_pred_list = self.predict_by_feat(voxel_dict, batch_input_metas)
        return seg_pred_list

    def predict_by_feat(self, voxel_dict: dict,
                        batch_input_metas: List[dict]) -> List[Tensor]:
        """Predict function.

        Args:
            voxel_dict (dict): The dict may contain `logits`,
                `point2voxel_map`.
            batch_input_metas (List[dict]): Meta information of a batch of
                samples.

        Returns:
            List[Tensor]: List of point-wise segmentation logits.
        """
        seg_logits = voxel_dict['logits']

        seg_pred_list = []
        coors = voxel_dict['coors']
        for batch_idx in range(len(batch_input_metas)):
            batch_mask = coors[:, 0] == batch_idx
            this_coors = coors[batch_mask, 1:]
            this_coors = this_coors.long()
            point_seg_predicts = seg_logits[batch_idx, this_coors[:, 0],
                                            this_coors[:, 1], this_coors[:, 2]]
            seg_pred_list.append(point_seg_predicts)
        return seg_pred_list
