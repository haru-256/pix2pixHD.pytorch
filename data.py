from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import random
from torchvision import transforms


class Cityscapes(Dataset):
    def __init__(self, path, phase="train"):
        """Cityscapes datasets

        Args:
            path (pathlib.Path): file path of cityscapes datasets.
            phase (str, optional): phase. Defaults to train.
        """

        if not path.is_absolute():
            self.abs_data_path = path.resolve()
        else:
            self.abs_data_path = path

        self.df = pd.read_csv(path)  # read csv file

    def __getitem__(self, idx):
        # read images
        real_images = Image.open(self.df["read_images_path"][idx]).convert("RGB")
        seg_maps = Image.open(self.df["segmentation_path"][idx])
        inst_map = Image.open(self.df["instance_path"][idx])

        # get params of preprocessing
        params = get_params(self.opt, real_images.size)
        # get transforms and apply
        transforms = get_transform(self.opt, params, real_images.size)
        real_tensor = transforms(real_images)

        transforms = get_transform(
            self.opt, params, method=Image.NEAREST, normalize=False
        )
        seg_tensor = (
            transforms(seg_maps) * 255.0
        )  # ToTensor converts values to range of [0, 1]
        inst_tensor = transforms(inst_map) * 255.0

        return real_tensor, seg_tensor, inst_tensor

    def __len__(self):
        return len(self.df)


def get_transform(self, opt, params, method=Image.BICUBIC, normalize=True):
    """get transforms for input tensor

    Args:
        opt (argparse): arguments of program
        params (dict): return of function 'get_params'
        method (Image.BICUBIC or Image.NEAREST): mthod for resizing images.
        normalize (bool): whether normalize image data
            by means=[0.5,0.5,0.5], std = [0.5,0.5,0.5].
    """
    transform_list = []
    if "resize" in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))
    elif "scale_width" in opt.resize_or_crop:
        transform_list.append(
            transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method))
        )

    if "crop" in opt.resize_or_crop:
        transform_list.append(
            transforms.Lambda(lambda img: __crop(img, params["crop_pos"], opt.fineSize))
        )

    if opt.resize_or_crop == "none":
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == "local":
            base *= 2 ** opt.n_local_enhancers
        transform_list.append(
            transforms.Lambda(lambda img: __make_power_2(img, base, method))
        )

    if not opt.no_flip:
        transform_list.append(
            transforms.Lambda(lambda img: __flip(img, params["flip"]))
        )

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [
            transforms.Normalize(
                (opt.mean, opt.mean, opt.mean), (opt.std, opt.std, opt.std)
            )
        ]
    return transforms.Compose(transform_list)


def get_params(opt, size):
    """get parameters of "crop_pos" and 'scale_width_and_crop'

    Args:
        opt (argparser): arguments of this program
        size (tuple): size of images

    Returns:
        params (dictionary): dictionary of parameters.
            Each key is "crop_pos", "scale_width_and_crop"
    """

    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == "resize_and_crop":
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == "scale_width_and_crop":
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    flip = random.random() > 0.5
    return {"crop_pos": (x, y), "flip": flip}


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
