import functools
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init


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
    n_downsample_global=3,
    n_blocks_global=9,
    n_local_enhancers=1,
    n_blocks_local=3,
    norm_type="instance",
    isAffine=True,
    gain=0.02,
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
    norm_type : str, optional
        Type of normalization layer (instance, batch) (the default is 'instance',
         which [default_description])
    isAffine : bool, optional
        whther apply affine in normalization layer (the default is True)
    gain : float, optional
        standard variation
    
    Return
    -------
    netG : torch.nn.Module
        Generator after initializie, according to g_type.
    """
    norm_layer = get_norm_layer(norm_type=norm_type)
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
    print("{} Generator", end="\n" + "=" * 50 + "\n")
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
        isActivation=False,
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

        isActivation : bool, optional
            whether apply Activation(ReLU) after add in ResBlock.
        
        """
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        # global generator model
        ngf_global = ngf * (2 ** n_local_enhancers)
        global_gen = GlobalGenerator(
            input_nc,
            output_nc,
            ngf_global,
            n_downsample_global,
            n_blocks_global,
            norm_layer,
            isActivation=isActivation,
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
                        isActivation=isActivation,
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
        isActivation=False,
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

        n_downsample : int, optional
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

        isActivation : bool, optional
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
                    isActivation=isActivation,
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
        isActivation=False,
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

        isActivation : bool, optional
            whether apply Activation(ReLU) after add.
        
        """
        super(ResnetBlock, self).__init__()
        self.isActivation = isActivation
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
        if self.isActivation:
            F.relu_(out)
            assert out.min() < 0, "output of ResBlock is not range of [0, inf)"
        return out
