import torch
import torch.nn as nn
import numpy as np


class Pix2PixHDModel(nn.Module):
    def __init__(self):
        return super().__init__()

    def forward(self, inputs):
        """forward

        Parameters
        ----------
        inputs : dict of nn.Tensor
            input to Generator. This is disctinary, each keys is "real_image",
            "label_map", "instance_map".

        Returns
        -------
        losses : dict of nn.Tensor (1,)
            Loss. Each keys is "g_gan", "g_fm", "g_p",
            "d_real", "d_fake".
        """

        return losses

    def encode_label_map(self, label_map):
        """encode label map to one-hot vector

        Parameters
        ----------
        label_map : nn.Tensor (N, C=1, H, W)
            label map. Each pixel is int and represents Label ID.

        Returns
        -------
        oneHot_label_map
            one-hot vector represents label map.
        """

        return oneHot_label_map

    def get_edge_map(self, instance_map):
        """make edge map and return edge map

        Parameters
        ----------
        instance_map : nn.Tensor (N, C=1, H, W)
            instance map. Each pixel is int and represents Label ID.

        Returns
        -------
        edge_map : nn.Tensor (N, C=1, H, W)
            Edge map. Each pixel is 1 or 0. 1 is boundary.
        """

        return edge_map
