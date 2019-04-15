import functools
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torchvision import models
import numpy as np


def weights_init(m, gain=0.02):
    """Initialize parameres.

    Args:
        m (torch.nn.Module): Module that means a layer.
        gain (float): standard variation

    """
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, gain)
        if hasattr(m, "bias") and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif class_name.find("BatchNorm2d") != -1:
        if hasattr(m, "weight"):
            init.normal_(m.weight.data, 1.0, gain)
        if hasattr(m, "bias"):
            init.constant_(m.bias.data, 0.0)


def get_norm_layer(norm_type="instance", isAffine=True):
    """get normalization layer according to Args(norm_type)

    Parameters
    ----------
    norm_type : str, optional
        Which normalization layer to choose.  (the default is 'instance')
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=isAffine)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=isAffine)
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def define_G(
    input_nc,
    output_nc,
    ngf,
    g_type,
    device,
    n_downsample_global=4,
    n_blocks_global=9,
    n_local_enhancers=1,
    n_blocks_local=3,
    norm_type="instance",
    isAffine=False,
    gain=0.02,
    use_relu=False,
):
    """define Generator

    Parameters
    ----------
    input_nc : int
        number of input channels

    output_nc : int
        number of output channels

    ngf : int
        number of gen filters in first conv layer

    device : torch.device
        device to send model.

    g_type : str
        Types of Generator (global or local)

    n_downsample_global : int, optional
        number of downsampling layers in netG (the default is 4,
        which [default_description])

    n_blocks_global : int, optional
        number of residual blocks in the global generator network (the default is 9,
         which [default_description])

    n_local_enhancers : int, optional
        number of local enhancers to use' (the default is 1,
         which [default_description])

    n_blocks_local : int, optional
         (the default is 3, which [default_description])

    norm_type : str, optional
        Type of normalization layer (instance, batch) (the default is 'instance',
         which [default_description])

    isAffine : bool, optional
        whther apply affine in normalization layer (the default is True)

    gain : float, optional
        standard division

    use_relu : bool, optional
        whether apply Activation(ReLU) after add in ResBlock.

    Return
    -------
    netG : torch.nn.Module
        Generator after initializie, according to g_type.
    """
    norm_layer = get_norm_layer(norm_type=norm_type, isAffine=isAffine)
    if g_type == "global":
        netG = GlobalGenerator(
            input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer
        )
    elif g_type == "local":
        netG = LocalEnhancer(
            input_nc,
            output_nc,
            ngf,
            n_downsample_global,
            n_blocks_global,
            n_local_enhancers,
            n_blocks_local,
            norm_layer,
        )
    else:
        raise ("generator not implemented!")
    print("{} Generator".format(g_type.capitalize()), end="\n" + "=" * 50 + "\n")
    print(netG, end="\n\n")

    netG.to(device)
    netG.apply(functools.partial(weights_init, gain=gain))

    return netG


class LocalEnhancer(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=32,
        n_downsample_global=4,
        n_blocks_global=9,
        n_local_enhancers=1,
        n_blocks_local=3,
        norm_layer=None,
        padding_type="reflect",
        use_relu=False,
    ):
        """Local Enhancer Generator

        Parameters
        ----------
        input_nc : int
            number of input channels

        output_nc : int
            number of output channels

        ngf : int
            number of gen filters in first conv layer

        n_downsample_global : int, optional
            number of downsampling layers in netG (the default is 3,
            which [default_description])

        n_blocks_global : int, optional
            number of residual blocks in the global generator network (the default is 9,
                which [default_description])

        n_local_enhancers : int, optional
            number of local enhancers to use' (the default is 1,
                which [default_description])

        n_blocks_local : int, optional
                (the default is 3, which [default_description])

        norm_layer : nn.norm, optional
            normalization layer any of the following 'instacne, batch'
             (the default is nn.BatchNorm2d, which [default_description])

        padding_type : str, optional
            padding type (the default is "reflect", which [default_description])

        use_relu : bool, optional
            whether apply Activation(ReLU) after add in ResBlock.

        """
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        # global generator model
        # print(str(n_local_enhancers), end="\n\n")
        ngf_global = ngf * (2 ** n_local_enhancers)
        global_gen = GlobalGenerator(
            input_nc,
            output_nc,
            ngf_global,
            n_downsample_global,
            n_blocks_global,
            norm_layer,
            use_relu=use_relu,
        ).global_gen
        global_gen = [
            global_gen[i] for i in range(len(global_gen) - 3)
        ]  # get rid of final convolution layers. 3 menas layes of [ReflactionPad, Conv2d, Tanh]
        self.global_gen = nn.Sequential(*global_gen)

        # local enhancer layers
        for n in range(1, n_local_enhancers + 1):
            # downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                norm_layer(ngf_global),
                nn.ReLU(True),
                nn.Conv2d(
                    ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1
                ),
                norm_layer(ngf_global * 2),
                nn.ReLU(True),
            ]
            # residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [
                    ResnetBlock(
                        ngf_global * 2,
                        padding_type=padding_type,
                        norm_layer=norm_layer,
                        use_relu=use_relu,
                    )
                ]

            # upsample
            model_upsample += [
                nn.ConvTranspose2d(
                    ngf_global * 2,
                    ngf_global,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                norm_layer(ngf_global),
                nn.ReLU(True),
            ]

            # final convolution
            if n == n_local_enhancers:
                model_upsample += [
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                    nn.Tanh(),
                ]

            setattr(self, "model" + str(n) + "_1", nn.Sequential(*model_downsample))
            setattr(self, "model" + str(n) + "_2", nn.Sequential(*model_upsample))

        # count_include_pad=Falseでaverage を計算する際に0 padしたところを計算に含めないようにする．
        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False
        )

    def forward(self, input_):
        """forward processing

        Args:
            input_ (nn.Tensor): segmentation label map(one-hot vector).
                this tensor is concatenated with edge_map and feature_vector

        Returns:
            output_prev (nn.Tensor): fake images.
        """

        # create input pyramid
        input_downsampled = [input_]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        # output at coarest level
        output_prev = self.model(input_downsampled[-1])
        # build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, "model" + str(n_local_enhancers) + "_1")
            model_upsample = getattr(self, "model" + str(n_local_enhancers) + "_2")
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


class GlobalGenerator(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        n_downsampling=4,
        n_blocks=9,
        norm_layer=None,
        padding_type="reflect",
        use_relu=False,
    ):
        """Bilut Global Genetrator.

        Parameters
        ----------
        input_nc : int
            number of input channels

        output_nc : int
            number of output channels

        ngf : int
            number of gen filters in first conv layer

        n_downsampling : int, optional
            number of downsampling layers in netG (the default is 4,
            which [default_description])

        n_blocks : int, optional
            number of residual blocks in the global generator network (the default is 9,
            which [default_description])

        norm_layer : nn.norm, optional
            normalization layer any of the following 'instacne, batch'
             (the default is nn.BatchNorm2d, which [default_description])

        padding_type : str, optional
            padding type (the default is "reflect", which [default_description])

        use_relu : bool, optional
            whether apply Activation(ReLU) after add in ResBolck.

        """
        super().__init__()
        assert norm_layer is not None, "must set norm_layer"
        activation = nn.ReLU(True)
        # c7s1-64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            activation,
        ]
        # downsample (dk)
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1
                ),
                norm_layer(ngf * mult * 2),
                activation,
            ]

        # resnet blocks (Rk)
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    use_relu=use_relu,
                )
            ]

        # upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                norm_layer(int(ngf * mult / 2)),
                activation,
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0, stride=1),
            nn.Tanh(),
        ]
        self.global_gen = nn.Sequential(*model)

    def forward(self, input_):
        return self.global_gen(input_)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        input_nc,
        padding_type,
        norm_layer,
        activation=nn.ReLU(True),
        use_dropout=False,
        use_relu=False,
    ):
        """built resnet block

        Parameters
        ----------
        input_nc : int
            number of input channels

        padding_type : str, optional
            padding type

        norm_layer : nn.norm, optional
            normalization layer any of the following 'instacne, batch'

        activation : nn.ReLU or nn.LeakyReLU, optional
            activation (the default is nn.ReLU(True), which [default_description])

        use_dropout : bool, optional
            whether use dropout in resblock (the default is False)

        use_relu : bool, optional
            whether apply Activation(ReLU) after add.

        """
        super(ResnetBlock, self).__init__()
        self.use_relu = use_relu
        self.conv_block = self.build_conv_block(
            input_nc, padding_type, norm_layer, activation, use_dropout
        )

    def build_conv_block(
        self, input_nc, padding_type, norm_layer, activation, use_dropout
    ):

        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(input_nc, input_nc, kernel_size=3, padding=p),
            norm_layer(input_nc),
            activation,
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(input_nc, input_nc, kernel_size=3, padding=p),
            norm_layer(input_nc),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        if self.use_relu:
            F.relu_(out)
            assert out.min() < 0, "output of ResBlock is not range of [0, inf)"
        return out


def define_D(
    input_nc,
    ndf,
    n_layers_D,
    device,
    norm_type="instance",
    use_sigmoid=False,
    num_D=3,
    getIntermFeat=True,
    gpu_ids=[],
    isAffine=False,
    gain=0.02,
):
    """bilut discriminator of pix2pixHD

    Parameters
    ----------
    input_nc : int
        number of input channels

    ndf : int
        number of dis  filters in first conv.

    n_layers : int
        nuber of dis layers

    device : torch.device
        device to send model.

    norm_type : str, optional
        Type of normalization layer (instance, batch) (the default is 'instance',
         which [default_description])

    use_sigmoid : bool, optional
        whether use sigmoid in the end of dis.

    num_D : int, optional
        number of discriminators

    getIntermFeat : bool, optional
        whether get intermediate features of discriminator to use FM loss.

    gpu_ids : list, optional
        [description] (the default is [], which [default_description])

    isAffine : bool, optional
        whther apply affine in normalization layer (the default is True)

    gain : float, optional
        standard variation

    Return
    ---------
    netD : nn.Module
        MultiDiscriminator after initializie
    """

    norm_layer = get_norm_layer(norm_type=norm_type, isAffine=isAffine)

    netD = MultiscaleDiscriminator(
        input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat
    )

    print("MultiscaleDiscriminator", end="\n" + "=" * 50 + "\n")
    print(netD, end="\n\n")

    netD.to(device)
    netD.apply(functools.partial(weights_init, gain=gain))

    return netD


class MultiscaleDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        use_sigmoid=False,
        num_D=3,
        getIntermFeat=True,
    ):
        """Multi Discriminator

        Parameters
        ----------
        input_nc : int
            number of input channels

        ndf : int, optional
            number of filters in first conv layer (the default is 64,
             which [default_description])

        n_layers : int, optional
            number of layers in single discriminator (the default is 3,
             which [default_description])

        norm_layer : [type], optional
            normalization layer (the default is nn.BatchNorm2d,
             which [default_description])

        use_sigmoid : bool, optional
            whether use sigmoid in the end of each discriminator.
             (the default is False, which [default_description])

        num_D : int, optional
            numbers of Discriminators (the default is 3, which [default_description])

        getIntermFeat : bool, optional
            whethere get intermediate features in each discriminator to use FM loss
             (the default is True, which [default_description])
        """

        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat
            )
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(
                        self,
                        "scale" + str(i) + "_layer" + str(j),
                        getattr(netD, "model" + str(j)),
                    )
            else:
                setattr(self, "layer" + str(i), netD.model)

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False
        )

    def singleD_forward(self, model, input_):
        """forward in single discriminator

        Parameters
        ----------
        model : nn.Module or list of nn.Module
            a single discrimnator. If self.getIntermFeat is True,
            If self.getIntermFeat is True, then model is a list and each element represents
            each layer of the single discriminator.

        input_ : nn.Tensor
            input tensor to single discriminator.

        Returns
        -------
        result : nn.Tensor or list of nn.Tensor
            output of the single discriminator.
            If self.getIntermFeat is True, then reslut is a list and each element represents
            each layer's output of the single discriminator.
        """

        if self.getIntermFeat:
            result = [input_]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input_)]

    def forward(self, input_):
        """forward

        Parameters
        ----------
        input_ : nn.Tensor
            input to MultiscaleDiscriminator

        Returns
        -------
        result : list of nn.Tensor or list of list of nn.Tensor
            output of MultiscaleDiscriminator. Each elemnet is output of a single
            discriminator If self.getIntermFeat is True, this is list of list of Tensor,
            each list is output of a single discriminator. And  each elements of inner
            list is intermediate features of discriminator.
        """

        num_D = self.num_D
        result = []
        input_downsampled = input_
        for i in range(num_D):
            if self.getIntermFeat:
                model = [
                    getattr(self, "scale" + str(num_D - 1 - i) + "_layer" + str(j))
                    for j in range(self.n_layers + 2)
                ]
            else:
                model = getattr(self, "layer" + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        use_sigmoid=False,
        getIntermFeat=True,
        slope=0.2,
    ):
        """single Discriminator with n layers

        Parameters
        ----------
        input_nc : int
            number of input channels

        ndf : int, optional
            number of dis  filters in first conv.

        n_layers : int, optional
            number of layers in discriminator (the default is 3,
             which [default_description])

        norm_layer : [type], optional
            Normalization Layer (the default is nn.BatchNorm2d,
             which [default_description])

        use_sigmoid : bool, optional
            whether use sigmoid in the end of discriminator (the default is False,
             which [default_description])

        getIntermFeat : bool, optional
            whether get intermediate features in discriminator to use FM loss
             (the default is False, which [default_description])

        slope : float, optional
            slope of LeakyReLU

        """

        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))

        sequence = [
            [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(slope, True),
            ]
        ]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                [
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                    norm_layer(nf),
                    nn.LeakyReLU(slope, True),
                ]
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(slope, True),
            ]
        ]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, "model" + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input_):
        """foeward of NLayerDiscriminator

        Parameters
        ----------
        input_ : Tensor, shape is (N, C, H, W)
            input tensor to single discriminator

        Returns
        -------
        result  : Tensor, shape is (N, 1, h, w)
            output of single Discriminator.
            If self.getIntermFeat is True, result is list of output of each layer.
        """

        if self.getIntermFeat:
            res = [input_]
            for n in range(self.n_layers + 2):
                model = getattr(self, "model" + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input_)


def define_E(
    input_nc,
    feat_num,
    nef,
    device,
    n_downsampling=4,
    norm_type="instance",
    isAffine=True,
    gain=0.02,
):
    """bilut Encoder

    Parameters
    ----------
    input_nc : int
        number of input channels.

    feat_num : int
        number of low-dim features (In other word, number of output channels).

    nef : int
        number of filters in first convlayers of Encoder.

    device : torch.device
        device to send model.

    n_downsample : int, optional
        number of downsampling layers expect for first conv layer(c7s1-k)
        (the default is 4, which [default_description])

    norm_type : str, optional
        Type of normalization layer (instance, batch) (the default is 'instance',
         which [default_description])

    isAffine : bool, optional
        whther apply affine in normalization layer (the default is True)

    gain : float, optional
        standard variation

    Return
    -------
    netE : torch.nn.Module
        Encoder after initializing

    """
    norm_layer = get_norm_layer(norm_type=norm_type, isAffine=isAffine)
    netE = Encoder(
        input_nc=input_nc,
        feat_num=feat_num,
        nef=nef,
        n_downsampling=n_downsampling,
        norm_layer=norm_layer,
    )
    print("Encoder", end="\n" + "=" * 50 + "\n")
    print(netE, end="\n\n")

    netE.to(device)
    netE.apply(functools.partial(weights_init, gain=gain))

    return netE


class Encoder(nn.Module):
    def __init__(
        self, input_nc, feat_num, nef=16, n_downsampling=4, norm_layer=nn.BatchNorm2d
    ):
        """Encoder

        Parameters
        ----------
        input_nc : int
            number of input channel

        feat_num : int
            number of low-dim features (In other word, number of output channels).

        nef : int, optional
            number of filters in first conv layer of Encoder
             (the default is 32, which [default_description])

        n_downsampling : int, optional
            number of downsampling layers in Encoder
             (the default is 4, which [default_description])

        norm_layer : [type], optional
            Normalization Layer (the default is nn.BatchNorm2d, which [default_description])

        """

        super(Encoder, self).__init__()
        self.output_nc = feat_num

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, nef, kernel_size=7, padding=0),
            norm_layer(nef),
            nn.ReLU(True),
        ]
        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    nef * mult, nef * mult * 2, kernel_size=3, stride=2, padding=1
                ),
                norm_layer(nef * mult * 2),
                nn.ReLU(True),
            ]

        # upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    nef * mult,
                    int(nef * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                norm_layer(int(nef * mult / 2)),
                nn.ReLU(True),
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(nef, self.output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input_, inst):
        """forward

        Parameters
        ----------
        input_ : nn.Tensor
            real images corresponding to segmentation map
        inst : nn.Tensor
            instance images corresponding to instance map

        Returns
        -------
        output_mean : nn.Tensor
            low dimensional features.
        """

        outputs = self.model(input_)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        # get instance id.
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input_.size()[0]):
                # get indices of pixel with instance ID: i
                indices = (inst[b] == int(i)).nonzero()  # n x 4
                # n is Number of pixels corresponding to i,
                # 4 is number of inst dimensions

                # make dimentional features with output_nc channels
                for j in range(self.output_nc):
                    output_ins = outputs[
                        indices[:, 0] + b,
                        indices[:, 1] + j,
                        indices[:, 2],
                        indices[:, 3],
                    ]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[
                        indices[:, 0] + b,
                        indices[:, 1] + j,
                        indices[:, 2],
                        indices[:, 3],
                    ] = mean_feat
        return outputs_mean


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        """VGG19

        Parameters
        ----------
        requires_grad : bool, optional
            whether need parameters gradient.
             (the default is False, which [default_description])

        """

        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        # fix parameters.
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input_):
        """forward

        Parameters
        ----------
        input_ : nn.Tensor shape is (N, C, H, W)
            input to VGG (Generated Image)

        Returns
        -------
        out : list of nn.Tensor
            each element is intermediate features of VGG to use Perceptual Loss.
        """

        h_relu1 = self.slice1(input_)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
