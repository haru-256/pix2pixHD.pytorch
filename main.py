import argparse
from training import Trainer, Updater
from data import Cityscapes
from torch.utils.data import DataLoader
from pix2pixHD_model import Pix2PixHDModel
import pathlib


def add_argument(parser):

    # add argument
    # base option of experiments
    parser.add_argument(
        "--name",
        type=str,
        default="label2city",
        help="name of the experiment. It decides where to store samples and models",
    )
    parser.add_argument("-s", "--seed", help="seed", type=int, required=True)
    parser.add_argument(
        "-n", "--number", help="the number of experiments.", type=int, required=True
    )
    parser.add_argument(
        "-e",
        "--epoch",
        help="the number of epoch, defalut value is 100",
        type=int,
        default=200,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="batch size. defalut value is 1",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-g",
        "--gpu_ids",
        help="specify gpu by this number. defalut value is 0,"
        " -1 is means don't use gpu",
        choices=[-1, 0, 1],
        type=int,
        default=1,
    )
    parser.add_argument(
        "--nThreads", default=2, type=int, help="# threads for loading data"
    )
    parser.add_argument(
        "--no_save_genImages", action="store_true", help="do not save generate images"
    )

    # for data
    parser.add_argument("--dataroot", type=str, default="./datasets/cityscapes/")
    parser.add_argument(
        "--loadSize", help="scale images to this size", type=int, default=1024
    )
    parser.add_argument(
        "--fineSize", type=int, default=512, help="then crop to this size"
    )
    parser.add_argument(
        "--label_num", type=int, default=35, help="# of input label channels"
    )
    parser.add_argument(
        "-m", "--mean", help="mean to use for noarmalization", type=float, default=0.5
    )
    parser.add_argument(
        "-std", "--std", help="std to use for noarmalization", type=float, default=0.5
    )
    parser.add_argument(
        "--resize_or_crop",
        type=str,
        default="scale_width",
        help="scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]",
    )
    parser.add_argument(
        "--no_flip",
        action="store_true",
        help="if specified, do not flip the images for data argumentation",
    )
    parser.add_argument(
        "--no_use_edge",
        action="store_true",
        help="if specified, do not use edge map as input to model",
    )

    # base option of models
    parser.add_argument(
        "--norm_type",
        help="specify normalization type. defalut value is `batch`,"
        "batch: Batch Normalization, instance: Instance Normalization, none: don't apply normalization",
        choices=["batch", "instance", "none"],
        default="batch",
    )
    parser.add_argument(
        "--input_nc", type=int, default=3, help="# of input image channels"
    )
    parser.add_argument(
        "--output_nc", type=int, default=3, help="# of output image channels"
    )
    parser.add_argument(
        "--isAffine", action="store_true", help="apply affine transformation."
    )

    # option of generator
    parser.add_argument(
        "--g_type", type=str, default="global", help="selects model to use for netG"
    )
    parser.add_argument(
        "--ngf", type=int, default=64, help="# of gen filters in first conv layer"
    )
    parser.add_argument(
        "--n_downsample_global",
        type=int,
        default=4,
        help="number of downsampling layers in netG",
    )
    parser.add_argument(
        "--n_blocks_global",
        type=int,
        default=9,
        help="number of residual blocks in the global generator network",
    )
    parser.add_argument(
        "--n_blocks_local",
        type=int,
        default=3,
        help="number of residual blocks in the local enhancer network",
    )
    parser.add_argument(
        "--n_local_enhancers",
        type=int,
        default=1,
        help="number of local enhancers to use",
    )
    parser.add_argument(
        "--niter_fix_global",
        type=int,
        default=0,
        help="number of epochs that we only train the outmost local enhancer",
    )
    parser.add_argument(
        "--use_relu",
        action="store_true",
        help="add ReLU Module after add processing in ResBlock",
    )

    # option of discriminator
    parser.add_argument(
        "--num_D", type=int, default=2, help="number of discriminators to use"
    )
    parser.add_argument(
        "--n_layers_D",
        type=int,
        default=3,
        help="only used if which_model_netD==n_layers",
    )
    parser.add_argument(
        "--ndf", type=int, default=64, help="# of discrim filters in first conv layer"
    )

    # option of encoder
    parser.add_argument(
        "--no_use_feature",
        action="store_true",
        help="do not use feature.",
    )
    parser.add_argument(
        "--feature_nc", type=int, default=3, help="vector length for encoded features"
    )
    parser.add_argument(
        "--load_features",
        action="store_true",
        help="if specified, load precomputed feature maps",
    )
    parser.add_argument(
        "--n_downsample_E",
        type=int,
        default=4,
        help="# of downsampling layers in encoder",
    )
    parser.add_argument(
        "--nef",
        type=int,
        default=16,
        help="# of encoder filters in the first conv layer",
    )
    parser.add_argument(
        "--n_clusters", type=int, default=10, help="number of clusters for features"
    )

    # option of optimizer
    parser.add_argument(
        "--niter_decay",
        type=int,
        default=100,
        help="# of iter to linearly decay learning rate to zero",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="momentum term of adam"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="initial learning rate for adam"
    )

    # option of loss
    parser.add_argument(
        "--lambda_feat",
        type=float,
        default=10.0,
        help="weight for feature matching loss",
    )
    parser.add_argument(
        "--lambda_perceptual",
        type=float,
        default=10.0,
        help="weight for perceptual loss using vgg",
    )
    parser.add_argument(
        "--no_fmLoss",
        action="store_true",
        help="if specified, do *not* use discriminator feature matching loss",
    )
    parser.add_argument(
        "--no_pLoss",
        action="store_true",
        help="if specified, do *not* use VGG feature matching loss",
    )
    parser.add_argument(
        "--no_lsgan",
        action="store_true",
        help="do *not* use least square GAN, if false, use vanilla GAN",
    )
    # hoge
    parser.add_argument(
        "-V", "--version", version="%(prog)s 1.0.0", action="version", default=False
    )
    return parser


if __name__ == "__main__":
    # make parser
    parser = argparse.ArgumentParser(
        prog="pix2pixHD",
        usage="`python main.py` for training",
        description="train pix2pixHD with city scapes",
        epilog="end",
        add_help=True,
    )
    opt = add_argument(parser).parse_args()
    print("=" * 60)
    print("arguments:", opt)
    print("=" * 60)

    # dataset
    dataroot = pathlib.Path(opt.dataroot)
    trainData_path = (dataroot / "train" / "cityscapes_train.csv").resolve()
    valData_path = (dataroot / "val" / "cityscapes_val.csv").resolve()
    train_dataset = Cityscapes(path=trainData_path, opt=opt)
    val_dataset = Cityscapes(path=valData_path, opt=opt, phase="val", valSize=9)
    print("Train Dataset Path :", trainData_path)
    print("Val Dataset Path :", valData_path)

    # make model
    pix2pixHD = Pix2PixHDModel(opt)

    # dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.nThreads,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        num_workers=opt.nThreads,
    )

    # updater
    updater = Updater(dataloader=train_dataloader, model=pix2pixHD)

    # trainer
    trainer = Trainer(updater, opt, val_dataloader=val_dataloader)

    # run
    log = trainer.run()

