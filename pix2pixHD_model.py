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
            device = torch.device("cuda:0")
        elif opt.gpu_id == 1:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cpu")

        self.device = torch.device()

        # define networks, respectively
        input_nc = opt.label_map
        if opt.use_feature:
            input_nc += opt.feature_nc
        if opt.use_edge:
            input_nc += 1
        self.netG = define_G(
            input_nc=input_nc,
            output_nc=opt.output_nc,
            ngf=opt.ngf,
            g_type=opt.g_type,
            device=device,
            isAffine=opt.isAffine,
            use_relu=opt.use_relu,
        )

        input_nc = opt.output_nc
        if opt.use_edge:
            input_nc += opt.label_num + 1
        else:
            input_nc += opt.label_num
        self.netD = define_D(
            input_nc=input_nc,
            ndf=opt.ndf,
            n_layers_D=opt.n_layers_D,
            device=self.device,
            isAffine=opt.isAffine,
        )

        self.netE = define_E(
            input_nc=opt.output_nc,
            feat_num=opt.feature_nc,
            nef=opt.nef,
            device=device,
            isAffine=opt.isAffine,
        )

        # define datasets, dataloader, dataloader

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

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.use_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(
            params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
        )
        if self.opt.verbose:
            print("------------ Now also finetuning global generator -----------")
