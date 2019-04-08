import torch
import torch.nn as nn
import numpy as np
from model import define_G, define_D, define_E


class Pix2PixHDModel(nn.Module):
    def __init__(self, opt):
        """Pix2PIxHD model

        Parameters
        ----------
        opt : ArgumentParsee
            option of this Model. e.g.)  gain, isAffine

        """
        super(Pix2PixHDModel, self).__init__()
        self.opt = opt

        if opt.gpu_id == 0:
            torch.device("cuda:0")
        elif opt.gpu_id == 1:
            torch.device("cuda:1")
        else:
            torch.device("cpu")

        self.device = torch.device()

        # define networks, respectively
        self.netG = define_G()
        if opt.use_edge:
            input_nc = opt.label_num + opt.input_nc + 1
        else:
            input_nc = opt.label_num + opt.input_nc
        self.netD = define_D(
            input_nc=input_nc,
            ndf=opt.ndf,
            n_layers_D=opt.n_layers_D,
            device=self.device,
        )
        self.netE = define_E(input_nc=opt.input_nc, feat_num=3, nef=opt.nef)

    def forward(self, inputs):
        """forward processing in one iteration.

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

