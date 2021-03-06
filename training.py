import pathlib
import json
import datetime
from collections import OrderedDict
import torch
from tqdm import tqdm
import pickle


class Trainer(object):
    def __init__(self, updater, opt, val_dataloader=None):
        """Trainer class

        Args:
            updater (Updater): Updater. This executes one epoch process,
                such as updating parameters
            opt (argparse): option of this program
            val_dataloader (dataloader): dataloader of validation datasets

        """
        # make directory
        self.opt = opt
        out = pathlib.Path(
            "{0}/result_{1}/result_{1}_{2}".format(opt.name, opt.number, opt.seed)
        ).resolve()
        for path in list(out.parents)[::-1]:
            if not path.exists():
                path.mkdir()
        if not out.exists():
            out.mkdir()

        # put arguments into file
        with open(out / "args.pkl", "wb") as f:
            pickle.dump(opt, f)
        with open(out / "args.txt", "w") as f:
            f.write(str(opt))

        # make dir to save model & image
        self.model_dir = out / "model"
        if not self.model_dir.exists():
            self.model_dir.mkdir()
        self.image_dir = out / "gen_images"
        if not self.image_dir.exists():
            self.image_dir.mkdir()

        self.epoch = self.opt.epoch
        self.updater = updater

        if val_dataloader:
            self.val_dataloader = val_dataloader

        torch.backends.cudnn.benchmark = True

    def run(self):
        """execute training loop.

        Returns:
            log (OrderDict): training log. This contains Generator loss and Discriminator Loss
        """
        since = datetime.datetime.now()
        # initialize log
        log = OrderedDict()

        # training loop
        epochs = tqdm(range(self.epoch), desc="Epoch", unit="epoch")
        for epoch in epochs:
            # itterater process
            losses, models, optimizers, schedulers = self.updater.update()

            # preserve train log & print train loss
            log["epoch_{}".format(epoch + 1)] = OrderedDict(
                train_dis_loss=losses["dis"], train_gen_loss=losses["gen"]
            )

            # save model & optimizers & losses by epoch
            torch.save(
                {
                    "epoch": epoch,
                    "dis_model_state_dict": models["dis"].state_dict(),
                    "gen_model_state_dict": models["gen"].state_dict(),
                    "en_model_state_dict": models["en"].state_dict(),
                    "dis_optim_state_dict": optimizers["dis"].state_dict(),
                    "gen_optim_state_dict": optimizers["gen"].state_dict(),
                    "dis_loss": losses["dis"],
                    "gen_loss": losses["gen"],
                    "dis_scheduler": schedulers["dis"],
                    "gen_scheduler": schedulers["gen"],
                },
                self.model_dir / "pix2pixHD_{}epoch.tar".format(epoch + 1),
            )

            # save generate images of validation
            if not self.opt.no_save_genImages:
                self.updater.model.save_gen_image(
                    epoch,
                    val_dataloader=self.val_dataloader,
                    root_dir4gen=self.image_dir,
                    device=self.updater.model.device,
                    mean=self.opt.mean,
                    std=self.opt.std,
                )

            # instead of only training the local enhancer, train the entire network after certain iterations
            if (self.opt.niter_fix_global != 0) and (
                self.epoch == self.opt.niter_fix_global
            ):
                self.updater.model.update_fixed_params()

            # linearly decay learning rate after certain iterations
            if epoch > self.opt.niter_decay:
                self.updater.model.scheduler_D.step()
                self.updater.model.scheduler_G.step()

            # print loss
            tqdm.write(
                "Epoch: {} GenLoss: {:.4f} DisLoss: {:.4f}".format(
                    epoch + 1, losses["gen"], losses["dis"]
                )
            )
            tqdm.write("-" * 60)

        time_elapsed = datetime.datetime.now() - since
        print("Training complete in {}".format(time_elapsed))
        # save log
        with open(self.out / "log.json", "w") as f:
            json.dump(log, f, indent=4, separators=(",", ": "))

        return log


class Updater(object):
    def __init__(self, dataloader, model):
        """Updater class.

       This class executes one epoch process, such as updating parameters.

        Args:
            dataloader (torch.utils.data.DataLoader):
                Dataloader that provides minibatch datasets.
            model (pix2pixHD.Pix2PixHD_model):
                pix2pixHD.Pix2PixHD_model. This model has to have optimizers and device.

        """
        self.dataloader = dataloader
        self.model = model
        self.device = self.model.device

    def update(self):
        """update parameters while one epoch.

        Returns:
            losses (dict): dictionary of loss of discriminator and generator.
                each key is dis and gen.
            models (dict): dictionary of model of discriminator, generator and encoder.
                each key is dis, gen and en.
            optimizers (dict): dictionary of optimizer of discriminator and generator.
                each key is dis and gen.
            schedulers (dict): dictionary of schedulers of discriminator and generator.
                each key is dis and gen.
        """
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        iteration = tqdm(self.dataloader, desc="Iteration", unit="iter")
        for real_image, label_map, instance_map in iteration:
            data_dict = {
                "real_image": real_image.to(self.device),
                "label_map": label_map.to(self.device),
                "instance_map": instance_map.to(self.device),
            }

            # forward process in Pix2PixHD
            loss_dict, gen_images = self.model(data_dict)

            # calculate final loss scalar
            loss_D = (loss_dict["d_real"] + loss_dict["d_fake"]) * 0.5
            loss_G = loss_dict["g_gan"] + loss_dict["g_fm"] + loss_dict["g_p"]
            assert (
                loss_D.dim() == 0 and loss_G.dim() == 0
            ), "Loss is not scalr. Got shape is : {}, {}".format(
                loss_D.shape, loss_G.shape
            )

            # backward Discrimintor
            self.model.optimizer_D.zero_grad()
            loss_D.backward()
            self.model.optimizer_D.step()

            # backward Generator
            self.model.optimizer_G.zero_grad()
            loss_G.backward()
            self.model.optimizer_G.step()

            epoch_loss_D += loss_D.detach()
            epoch_loss_G += loss_G.detach()

        losses = {
            "dis": epoch_loss_D.item() / len(self.dataloader),
            "gen": epoch_loss_G.item() / len(self.dataloader),
        }
        models = {"dis": self.model.netD, "gen": self.model.netG, "en": self.model.netE}
        optimizers = {"dis": self.model.optimizer_D, "gen": self.model.optimizer_G}
        schedulers = {"dis": self.model.scheduler_D, "gen": self.model.scheduler_G}

        return losses, models, optimizers, schedulers

