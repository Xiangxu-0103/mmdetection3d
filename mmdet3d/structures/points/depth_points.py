# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import numpy as np
import torch

from mmdet3d.structures import Coord3DMode
from .base_points import BasePoints


class DepthPoints(BasePoints):
    """Points of instances in DEPTH coordinates.

    Args:
        tensor (torch.Tensor or np.ndarray or Sequence):
            A N x points_dim matrix.
        points_dim (int): Number of the dimension of a point.
            Each row is (x, y, z). Defaults to 3.
        attribute_dims (dict, optional): Dictionary to indicate the
            meaning of extra dimension. Defaults to None.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x points_dim.
        points_dim (int): Integer indicating the dimension of a point.
            Each row is (x, y, z, ...).
        attribute_dims (dict): Dictionary to indicate the meaning of extra
            dimension. Defaults to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self,
                 tensor: Union[torch.Tensor, np.ndarray, Sequence],
                 points_dim: int = 3,
                 attribute_dims: Optional[dict] = None) -> None:
        super(DepthPoints, self).__init__(
            tensor, points_dim=points_dim, attribute_dims=attribute_dims)
        self.rotation_axis = 2

    def flip(self, bev_direction: str = 'horizontal') -> None:
        """Flip the points along given BEV direction.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
        """
        if bev_direction == 'horizontal':
            self.tensor[:, 0] = -self.tensor[:, 0]
        elif bev_direction == 'vertical':
            self.tensor[:, 1] = -self.tensor[:, 1]

    def convert_to(
        self,
        dst: Coord3DMode,
        rt_mat: Optional[Union[np.ndarray,
                               torch.Tensor]] = None) -> BasePoints:
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`CoordMode`): The target Point mode.
            rt_mat (np.ndarray or torch.Tensor, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BasePoints`: The converted point of the same type
            in the `dst` mode.
        """
        return Coord3DMode.convert_point(
            point=self, src=Coord3DMode.DEPTH, dst=dst, rt_mat=rt_mat)
