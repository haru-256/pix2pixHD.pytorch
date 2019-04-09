import torch
import torch.nn as nn
from utils import LinearDecayLR
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
            self.device = torch.device("cuda:0")
        elif opt.gpu_id == 1:
            self.device = torch.device("cuda:1")
        else:
            self.device = torch.device("cpu")

        # define networks respectively
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
            device=self.device,
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
            device=self.device,
            isAffine=opt.isAffine,
        )

        # define optimizer respectively
        # initialize optimizer G&E
        # if opt.niter_fix_global is True, fix parameters in Global Generator
        if opt.niter_fix_global > 0:
            finetune_list = set()

            params = []
            for key, value in self.netG.named_parameters():
                if key.startswith("model" + str(opt.n_local_enhancers)):
                    params += [value]
                    finetune_list.add(key.split(".")[0])
            print(
                "------------- Only training the local enhancer network (for %d epochs) ------------"
                % opt.niter_fix_global
            )
            print("The layers that are finetuned are ", sorted(finetune_list))
        else:
            params = list(self.netG.parameters())
        if self.use_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
        self.scheduler_G = LinearDecayLR(self.optimizer_G, niter_decay=opt.niter_decay)

        # initialize optimizer D
        # optimizer D
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
        )
        self.scheduler_D = LinearDecayLR(self.optimizer_D, niter_decay=opt.niter_decay)

    def data2device(self, data_dict):
        """migrate data to device(e.g. cuda)

        Args:
            data_dict (dict): each keys is "real_image", "label_map", "instance_map".

        Returns:
            data_dict (dict): arguments(data_dict) after migrating.
        """
        for key, data in data_dict.items():
            if data is None:
                continue
            data_dict[key] = data.to(self.device)
        return data_dict

    def forward(self, data_dict):
        """forward processing in one iteration.

        Parameters
        ----------
        data_dict : dict of nn.Tensor
            input to Generator or Discriminator. This is disctinary, each keys is
            "real_image", "label_map", "instance_map".

        Returns
        -------
        losses : dict of nn.Tensor (1,)
            Loss. Each keys is "g_gan", "g_fm", "g_p",
            "d_real", "d_fake".
        """
        # data migrate device
        data_dict = self.data2device(data_dict)
        # encode label_map
        data_dict["label_map"] = self.encode_label_map(
            data_dict["label_map"], self.opt.label_num
        )
        # get edge_map
        data_dict["edge_map"] = self.get_edge_map(data_dict["instance_map"])

        # Generate fake image

        feat_vector = netE()
        gen_inputs = torch.cat((data_dict["label_map"], data_dict["edge_map"]))
        self.netG()

        return losses

    def encode_label_map(self, label_map, n_class):
        """encode label map to one-hot vector

        Parameters
        ----------
        label_map : nn.Tensor (N, C=1, H, W)
            label map. Each pixel is int and represents Label ID.

        n_class : int
            number of classes in label map.

        Returns
        -------
        oneHot_label_map
            one-hot vector label map.
        """

        oneHot_label_map = torch.FloatTensor(
            label_map.size(0), n_class, label_map.size(2), label_map.size(3)
        ).zero_()
        oneHot_label_map = oneHot_label_map.scatter_(1, label_map, 1)
        assert (0.0 <= oneHot_label_map).all() and (
            oneHot_label_map <= 1.0
        ).all(), "failed at encoding to make one-hot vector: range is [{}, {}]".format(
            oneHot_label_map.min(), oneHot_label_map.max()
        )
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
        edge_map = torch.ByteTensor(t.size()).zero_().to(self.device)
        edge_map[:, :, :, 1:] = edge_map[:, :, :, 1:] | (
            instance_map[:, :, :, 1:] != instance_map[:, :, :, :-1]
        )
        edge_map[:, :, :, :-1] = edge_map[:, :, :, :-1] | (
            instance_map[:, :, :, 1:] != instance_map[:, :, :, :-1]
        )
        edge_map[:, :, 1:, :] = edge_map[:, :, 1:, :] | (
            instance_map[:, :, 1:, :] != instance_map[:, :, :-1, :]
        )
        edge_map[:, :, :-1, :] = edge_map[:, :, :-1, :] | (
            instance_map[:, :, 1:, :] != instance_map[:, :, :-1, :]
        )
        if self.opt.data_type == 16:
            return edge_map.half()
        else:
            return edge_map.float()

        return edge_map

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.use_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(
            params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
        )
        self.scheduler_G = LinearDecayLR(
            self.optimizer_G, niter_decay=self.opt.niter_decay
        )
        if self.opt.verbose:
            print("------------ Now also finetuning global generator -----------")
