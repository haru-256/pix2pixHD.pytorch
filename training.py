import pathlib
import json
import datetime
from collections import OrderedDict
import torch
from fastprogress import master_bar, progress_bar


class Trainer(object):
    def __init__(self, updater, opt):
        """Trainer class

        Args:
            updater (Updater): Updater. This executes one epoch process,
                such as updating parameters
            out (string): Dir path to save experiment results.
                e.g.) "result_{number}/result_{number}_{seed}"
            epoch (int): number of epochs.

        """
        # make directory
        self.opt = opt
        out = pathlib.Path(
            "{0}/result_{1}/result_{1}_{2}".format(opt.dataset, opt.number, opt.seed)
        ).resolve()
        for path in list(out.parents)[::-1]:
            if not path.exists():
                path.mkdir()
        if not out.exists():
            out.mkdir()

        # put arguments into file
        with open(out / "args.txt", "w") as f:
            f.write(str(opt))
        print("arguments:", opt)

        # make dir to save model & image
        self.model_dir = out / "model"
        if not self.model_dir.exists():
            self.model_dir.mkdir()
        self.image_dir = out / "gen_images"
        if not self.image_dir.exists():
            self.image_dir.mkdir()

        self.epoch = self.opt.epoch
        self.updater = updater

    def run(self):
        """execute training loop.
        """
        since = datetime.datetime.now()
        # initialize log
        log = OrderedDict()

        # training loop
        mb = master_bar(self.epoch)
        for epoch in mb:
            # itterater process
            losses, models, optimizers, schedulers = self.updater.update(mb)

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
                    "dis_optim_state_dict": optimizers["dis"].state_dict(),
                    "gen_optim_state_dict": optimizers["gen"].state_dict(),
                    "dis_loss": losses["dis"],
                    "gen_loss": losses["gen"],
                    "dis_scheduler": schedulers["dis"],
                    "gen_scheduler": schedulers["gen"],
                },
                self.model_dir / "pix2pixHD_{}epoch.tar".format(epoch + 1),
            )

            # instead of only training the local enhancer, train the entire network after certain iterations
            if (self.opt.niter_fix_global != 0) and (
                self.epoch == self.opt.niter_fix_global
            ):
                self.model.update_fixed_params()

            # linearly decay learning rate after certain iterations
            if epoch > self.opt.niter:
                self.model.scheduler_D.step()
                self.model.scheduler_G.step()

            mb.first_bar.comment = "epoch stat"

            # print loss
            mb.write(
                "Epoch: {} GenLoss: {:.4f} DisLoss: {:.4f}".format(
                    epoch + 1, losses["gen"], losses["dis"]
                )
            )

        time_elapsed = datetime.datetime.now() - since
        print("Training complete in {}".format(time_elapsed))
        # save log
        with open(self.out / "log.json", "w") as f:
            json.dump(log, f, indent=4, separators=(",", ": "))

        return log


class Updater(object):
    def __init__(self, dataloader, model, device, **kwargs):
        """Updater class.

       This class executes one epoch process, such as updating parameters.

        Args:
            dataloader (torch.utils.data.DataLoader):
                Dataloader that provides minibatch datasets.
            model (pix2pixHD.Pix2PixHD_model):
                pix2pixHD.Pix2PixHD_model. This model has to have optimizers.
            device (torch.device): device object of data destination.

        Returns:
            losses (dict): dictionary of loss of discriminator and generator.
                each key is dis and gen.
            models (dict): dictionary of model of discriminator, generator and encoder.
                each key is dis, gen and en.
            optimizers (dict): dictionary of optimizer of discriminator and generator.
        """
        self.dataloader = dataloader
        self.model = model
        self.device = device

    def update(self, master_bar):
        """update parameters while one epoch.

        Args:
            master_bar (fastprogress.master_bar): master_bar that has epoch.

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
        for real_image, labe_map, instance_map in progress_bar(
            self.dataloader, parent=master_bar
        ):
            data_dict = {
                "real_image": real_image.to(self.device),
                "labelmap": labe_map.to(self.device),
                "instance_map": instance_map.to(self.device),
            }
            # zeroes the gradient buffers of all parameters.
            self.model.optimizer_G.zero_grad()
            self.model.optimizer_D.zero_grad()

            # forward process in Pix2PixHD
            loss_dict, gen_images = self.model(data_dict)

            # calculate final loss scalar
            loss_D = loss_dict["d_real"] + loss_dict["d_fake"]
            loss_G = loss_dict["g_gan"] + loss_dict["g_fm"] + loss_dict["g_p"]

            # backward pass
            loss_D.backward()
            loss_G.backward()
            # update parameters, respectively
            self.model.optimizer_D.step()
            self.model.optimizer_G.step()

            epoch_loss_D += loss_D
            epoch_loss_G += loss_G

            master_bar.child.comment = f"iteration stat"

        losses = {
            "dis": epoch_loss_D.item() / len(self.dataloader),
            "gen": epoch_loss_G.item() / len(self.dataloader),
        }
        models = {"dis": self.model.netD, "gen": self.model.netG, "en": self.model.netE}
        optimizers = {"dis": self.model.optimizer_D, "gen": self.model.optimizer_G}
        schedulers = {"dis": self.model.scheduler_D, "gen": self.model.scheduler_G}

        return losses, models, optimizers, schedulers

