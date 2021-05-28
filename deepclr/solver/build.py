import torch

from ..config.config import Config
from . import optimizers
from . import schedulers
from .schedulers import LRScheduler


def make_optimizer(cfg: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create optimizer from config and model."""
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.optimizer.base_lr
        weight_decay = cfg.optimizer.weight_decay
        if 'bias' in key:
            lr = cfg.optimizer.base_lr * cfg.optimizer.bias_lr_factor
            weight_decay = cfg.optimizer.weight_decay_bias
        params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]

    class_ = getattr(optimizers, cfg.optimizer.name)
    optimizer = class_(params, **cfg.optimizer.params)
    return optimizer


def make_scheduler(cfg: Config, optimizer: torch.optim.Optimizer) -> LRScheduler:
    """Create scheduler from config and optimizer."""
    class_ = getattr(schedulers, cfg.scheduler.name)
    scheduler = class_(optimizer, **cfg.scheduler.params)
    return scheduler
