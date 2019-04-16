from copy import deepcopy
from tqdm import tqdm
import torch


class LinearDecayLR(object):
    def __init__(self, optimizer, last_epoch=-1, niter_decay=100):
        """Linear Decay scheduler.

        This does not support the optimizer which has multiple groups of parameters.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            lr_lambda ([type]): [description]
            last_epoch (int, optional): Defaults to -1. [description]
            niter_decay (int, optional): Defaults to 100. [description]

        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.niter_decay = niter_decay
        self.initial_lrs = list(map(lambda group: group["lr"], optimizer.param_groups))
        self.old_lrs = deepcopy(self.initial_lrs)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self, epoch=None):
        """a step of learning rate scheduler
            epoch (int, optional): Defaults to None.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

    def decay_lr(self, old_lr, initial_lr):
        """function that returns next step learning rate.

        Parameter
        -----------
        old_lr : float
            current learning rate

        initial_lr : float
            initia learning rate

        Return
        -----------
        next_lr : float
            netx step learning rate.
        """
        lrd = initial_lr / self.niter_decay
        return old_lr - lrd

    def get_lr(self):
        """get next learning rate

        Returns:
            next_lr (list): each elemnet is next learning rate
        """
        next_lr = []
        for old_lr, initial_lr in zip(self.old_lrs, self.initial_lrs):
            next_lr.append(self.decay_lr(old_lr, initial_lr))
        tqdm.write("=" * 60)
        tqdm.write("Update learning rate: {} -> {}".format(self.old_lrs, next_lr))
        tqdm.write("=" * 60)

        self.old_lrs = next_lr

        return next_lr

