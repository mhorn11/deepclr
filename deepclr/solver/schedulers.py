import abc
from typing import Any, Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, CyclicLR


class LRScheduler(_LRScheduler, metaclass=abc.ABCMeta):
    """Abstract base class for custom learning rate schedulers."""
    def __init__(self, optimizer):
        super().__init__(optimizer)

    @abc.abstractmethod
    def get_lr(self) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, metric: Optional[float] = None) -> None:
        super().step()


class CyclicLRWithFlatAndCosineAnnealing(LRScheduler):
    """First cyclic learning rate then flat learning rate and at the end cosine annealing."""
    def __init__(self, optimizer: torch.optim.Optimizer, cyclic_iterations: int, flat_iterations: int,
                 annealing_iterations: int, base_lr: float, *args: Any, **kwargs: Any):
        self.is_initialized = False

        self.cyclic_lr = CyclicLR(optimizer, base_lr, *args, **kwargs)
        self.annealing = CosineAnnealingLR(optimizer, T_max=annealing_iterations, eta_min=0.0)

        self.cyclic_base_lr = base_lr
        self.cyclic_iterations = cyclic_iterations
        self.flat_iterations = flat_iterations
        self.annealing_iterations = annealing_iterations

        super().__init__(optimizer)

        self.annealing.base_lrs = [self.cyclic_base_lr for _ in range(len(self.base_lrs))]  # type: ignore
        self.is_initialized = True

    def get_lr(self) -> Any:
        last_epoch = self.last_epoch  # type: ignore
        if last_epoch < self.cyclic_iterations:
            return self.cyclic_lr.get_last_lr()
        elif last_epoch < self.cyclic_iterations + self.flat_iterations:
            return [self.cyclic_base_lr for _ in range(len(self.base_lrs))]  # type: ignore
        else:
            return self.annealing.get_last_lr()

    def step(self, metric: Optional[float] = None) -> None:
        last_epoch = self.last_epoch  # type: ignore
        if not self.is_initialized:
            super().step(metric=metric)

        elif last_epoch < self.cyclic_iterations:
            self.cyclic_lr.step()
            super().step(metric=metric)

        elif last_epoch < self.cyclic_iterations + self.flat_iterations:
            super().step(metric=metric)

        elif last_epoch < self.cyclic_iterations + self.flat_iterations + self.annealing_iterations:
            self.annealing.step()
            super().step(metric=metric)
