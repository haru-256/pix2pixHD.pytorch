import torch
import torch.nn as nn
from utils import LinearDecayLR
from model import define_G, define_D, define_E
from loss import GANLoss, PerceptualLoss, FMLoss
import torchvision
import math
import matplotlib.pyplot as plt
from tqdm import tqdm


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

        if opt.gpu_ids == 0:
            self.device = torch.device("cuda:0")
        elif opt.gpu_ids == 1:
            self.device = torch.device("cuda:1")
        else:
            self.device = torch.device("cpu")

        # define networks respectively
        input_nc = opt.label_num
        if not opt.no_use_feature:
            input_nc += opt.feature_nc
        if not opt.no_use_edge:
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
        if not opt.no_use_edge:
            input_nc += opt.label_num + 1
        else:
            input_nc += opt.label_num
        self.netD = define_D(
            input_nc=input_nc,
            ndf=opt.ndf,
            n_layers_D=opt.n_layers_D,
            device=self.device,
            isAffine=opt.isAffine,
            num_D=opt.num_D,
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
        if not self.opt.no_use_feature:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
        self.scheduler_G = LinearDecayLR(self.optimizer_G, niter_decay=opt.niter_decay)

        # initialize optimizer D
        # optimizer D
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
        )
        self.scheduler_D = LinearDecayLR(self.optimizer_D, niter_decay=opt.niter_decay)

        # defin loss functions
        if opt.gpu_ids == 0 or opt.gpu_ids == 1:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.FloatTensor

        self.criterionGAN = GANLoss(
            self.device, use_lsgan=not opt.no_lsgan, tensor=self.Tensor
        )
        if not self.opt.no_fmLoss:
            self.criterionFM = FMLoss(
                num_D=opt.num_D, n_layers=opt.n_layers_D, lambda_feat=opt.lambda_feat
            )
        if not self.opt.no_pLoss:
            self.criterionP = PerceptualLoss(
                self.device, lambda_perceptual=opt.lambda_perceptual
            )

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

        losses = {"g_gan": 0, "g_fm": 0, "g_p": 0, "d_real": 0, "d_fake": 0}

        # Generate fake image
        feat_vector = self.netE(
            input_=data_dict["real_image"], inst=data_dict["instance_map"]
        )
        gen_inputs = torch.cat(
            (data_dict["label_map"], data_dict["edge_map"], feat_vector), dim=1
        )
        fake_images = self.netG(gen_inputs)
        assert (
            (-1 <= fake_images) * (fake_images <= 1)
        ).all(), "input data to discriminator range is not from -1 to 1. Got: {}".format(
            (fake_images.min(), fake_images.max())
        )

        # Discrimintor Loss = GANLoss(fake) + GANLoss(real)
        dis_inputs_fake = torch.cat(
            (fake_images.detach(), data_dict["label_map"], data_dict["edge_map"]), dim=1
        )
        dis_inputs_real = torch.cat(
            (data_dict["real_image"], data_dict["label_map"], data_dict["edge_map"]),
            dim=1,
        )
        pred_fake = self.netD(dis_inputs_fake)
        pred_real = self.netD(dis_inputs_real)

        losses["d_fake"] = self.criterionGAN(pred_fake, target_is_real=False)
        losses["d_real"] = self.criterionGAN(pred_real, target_is_real=True)

        # Generator Loss = GANLoss(fake passability loss) + FMLoss + PerceptualLoss
        dis_inputs = torch.cat(
            (fake_images, data_dict["label_map"], data_dict["edge_map"]), dim=1
        )
        pred_fake = self.netD(dis_inputs)
        losses["g_gan"] = self.criterionGAN(pred_fake, target_is_real=True)
        if not self.opt.no_fmLoss:
            losses["g_fm"] = self.criterionFM(pred_fake, pred_real)
        if not self.opt.no_pLoss:
            losses["g_p"] = self.criterionP(fake_images, data_dict["real_image"])

        return losses, fake_images

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

        oneHot_label_map = (
            torch.FloatTensor(
                label_map.size(0), n_class, label_map.size(2), label_map.size(3)
            )
            .zero_()
            .to(self.device)
        )
        oneHot_label_map = oneHot_label_map.scatter_(1, label_map.long(), 1)
        assert (0.0 <= oneHot_label_map).all() and (
            oneHot_label_map <= 1.0
        ).all(), "failed at encoding to make one-hot vector: range is [{}, {}]".format(
            oneHot_label_map.min(), oneHot_label_map.max()
        )
        assert (
            torch.unique(oneHot_label_map).numel() == 2
        ), "label map is not one-hot vector"
        return oneHot_label_map

    def get_edge_map(self, instance_map):
        """make edge map and return edge map
data_dict
        Parameters
        ----------
        instance_map : nn.Tensor (N, C=1, H, W)
            instance map. Each pixel is int and represents Label ID.

        Returns
        -------
        edge_map : nn.Tensor (N, C=1, H, W)
            Edge map. Each pixel is 1 or 0. 1 is boundary.
        """
        edge_map = torch.ByteTensor(instance_map.size()).zero_().to(self.device)
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
        return edge_map.float()

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
            tqdm.write("------------ Now also finetuning global generator -----------")

    def save_gen_image(
        self, epoch, val_dataloader, root_dir4gen=None, device=None, mean=0.5, std=0.5
    ):
        """
        save generator images

        Parmameters
        -------------------
        epoch: int
            number of epochs
        val_dataloader: torch.utils.data.DataLoader
            dataloader for val
        root_dir4gen: pathlib.Path
            path to save generate images directory
        device: torch.device
        mean: float
            mean that is used to normalize inputs data.
        std: float
            std that is used to normalize inputs data.
        """
        # self.netG.train()  # apply Dropout and BatchNorm during inference as well
        # self.netE.train()
        with torch.no_grad():
            for real_image, label_map, inst_map in val_dataloader:
                data_dict = {
                    "real_image": real_image.to(self.device),
                    "label_map": label_map.to(self.device),
                    "instance_map": inst_map.to(self.device),
                }
                self.data2device(data_dict)
                # encode label_map
                data_dict["label_map"] = self.encode_label_map(
                    data_dict["label_map"], self.opt.label_num
                )
                # get edge_map
                data_dict["edge_map"] = self.get_edge_map(data_dict["instance_map"])

                # Generate fake image
                feat_vector = self.netE(
                    input_=data_dict["real_image"], inst=data_dict["instance_map"]
                )
                gen_inputs = torch.cat(
                    (data_dict["label_map"], data_dict["edge_map"], feat_vector), dim=1
                )
                fake_images = self.netG(gen_inputs).cpu()
                assert (
                    (-1 <= fake_images) * (fake_images <= 1)
                ).all(), "input data to discriminator range is not from -1 to 1. Got: {}".format(
                    (fake_images.min(), fake_images.max())
                )

        total = fake_images.shape[0]
        ncol = int(math.sqrt(total))
        nrow = math.ceil(float(total) / ncol)
        images = torchvision.utils.make_grid(
            fake_images, normalize=False, nrow=nrow, padding=1
        )
        images = images * std + mean
        plt.figure(figsize=(12, 9))
        # plt.figure(figsize=(16, 12))
        plt.imshow(images.numpy().transpose(1, 2, 0))
        plt.axis("off")
        plt.title("Epoch: {}".format(epoch + 1))
        plt.tight_layout()
        plt.savefig(
            root_dir4gen / "epoch{:0>4}.png".format(epoch),
            bbox_inches="tight",
            pad_inches=0.05,
            dpi=200,
        )
        plt.close()

