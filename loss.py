import torch.nn as nn
import torch
from model import Vgg19


class GANLoss(nn.Module):
    def __init__(
        self,
        device,
        use_lsgan=True,
        target_real_label=1.0,
        target_fake_label=0.0,
        tensor=torch.FloatTensor,
    ):
        """GAN Loss

        Parameters
        ----------
        device : torch.device
            device object

        use_lsgan : bool, optional
            whether criterion is lsgan or vanilla gan
            (the default is True, which [default_description])

        target_real_label : float, optional
            target tensor correponding to real image
            (the default is 1.0, which [default_description])

        target_fake_label : float, optional
            target tensor corresponding to fake image
            (the default is 0.0, which [default_description])

        tensor : torch.FloatTensor, torch.HalfTensor, optional
            dtype of target tensor.
            (the default is torch.FloatTensor, which [default_description])

        """

        super(GANLoss, self).__init__()
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        self.real_label = None
        self.fake_label = None
        self.Tensor = tensor
        self.device = device

        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input_, target_is_real):
        """get target label

        Parameters
        ----------
        input_ : nn.Tensor
            output of a single discriminator.

        target_is_real : bool
            whether target label is 1.0.

        Returns
        -------
        target_tensor : nn.Tensor
            target tensor according target_is_real
        """

        if target_is_real:
            create_label = (self.real_label is None) or (
                self.real_label.numel() != input_.numel()
            )
            if create_label:
                self.real_label = self.Tensor(input_.size()).fill_(
                    self.target_real_label
                )
                self.real_label.requires_grad = False
            target_tensor = self.real_label
            assert (
                target_tensor.unique().item() == self.target_real_label
            ), "target_tensor does not match target_real_label : {}".format(
                self.target_real_label
            )
        else:
            create_label = (self.fake_label is None) or (
                self.fake_label.numel() != input_.numel()
            )
            if create_label:
                self.fake_label = self.Tensor(input_.size()).fill_(self.target_fake_label)
                self.fake_label.requires_grad = False
            target_tensor = self.fake_label
            assert (
                target_tensor.unique().item() == self.target_fake_label
            ), "target_tensor does not match target_fake_label : {}".format(
                self.target_fake_label
            )

        return target_tensor.to(self.device)

    def forward(self, input_, target_is_real):
        """forward

        Parameters
        ----------
        input : list of nn.Tensor or list of list of nn.Tensor
            output of MultiscalesDisciriminator. If use feature matching loss,
            this is list of list of nn.Tensor, each list is output of
            a single discriminator. And  each elements of inner
            list is intermediate features of discriminator.

        target_is_real : bool
            whether target label is 1

        Returns
        -------
        loss
            GAN(LSGAN) loss
        """

        # judge if discriminator num is 1, or multi.
        if isinstance(input_[0], list):
            # multiscale discriminator.
            loss = 0
            for input_i in input_:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input_[-1], target_is_real)
            return self.loss(input_[-1], target_tensor)


class PerceptualLoss(nn.Module):
    def __init__(self, device, lambda_perceptual=10.0):
        """Perceptual Loss using VGG19

        Parameters
        ----------
        device : torch.device
            gpu or cpu.

        lambda_perceptual : float
            weight of perceptual loss.

        """

        super(PerceptualLoss, self).__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        """forward

        Parameters
        ----------
        x : nn.Tensor (N, C, H, W)
            Generated Image

        y : nn.Tensor (N, C, H, W)
            Real Image

        Returns
        -------
        loss : nn.Tensor (1,)
            Perceptual Loss using VGG19.
        """

        assert (
            x.size(1) == 3 and y.size(1) == 3
        ), "inputs to forward method is not images. Got shape:{}, {}".format(
            x.size(), y.size()
        )
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class FMLoss(nn.Module):
    def __init__(
        self, num_D=3, n_layers=3, feat_weights=4.0, d_weights=1.0, lambda_feat=10.0
    ):
        """Feature Matching Loss via Discriminator

        Parameters
        ----------
        num_D : int
            number of discriminators.

        n_layers : int
            number of layers of discriminators.

        feat_weights : float, optional
            features weights (the default is 4.0, which [default_description])

        d_weights : float, optional
            Discriminator weights (the default is 1.0, which [default_description])

        lambda_feat : float, optional
            weight of features matching loss.
        """
        super(FMLoss, self).__init__()

        self.feat_weights = feat_weights / (n_layers + 1)
        self.d_weights = d_weights / num_D
        self.num_D = num_D
        self.lambda_feat = lambda_feat
        self.criterion = nn.L1Loss()

    def forward(self, pred_fake, pred_real):
        """calculate loss of Feature Matching Loss.

        Parameters
        ----------
        pred_fake : list of list of nn.Tensor
            result of MultiscaleDiscriminator for fake images. each element represents
            each layer's output of the single discriminator.

        pred_real : list of list of nn.Tensor
            result of MultiscaleDiscriminator for real images. each element represents
            each layer's output of the single discriminator.

        Returns
        -------
        loss : nn.Tensor (1, )
            Feature Matching Loss.
        """

        loss = 0
        for i in range(self.num_D):
            for j in range(len(pred_fake[i]) - 1):
                loss += (
                    self.d_weights
                    * self.feat_weights
                    * self.criterion(pred_fake[i][j], pred_real[i][j].detach())
                    * self.lambda_feat
                )

        return loss
