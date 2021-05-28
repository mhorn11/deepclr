from typing import Any, Callable, Optional, Tuple

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
import torch

from ..utils.metrics import MetricFunction


def _no_transform(x: Any) -> Any:
    """Default output transform."""
    return x


class LossFn(Metric):
    def __init__(self, loss_fn: MetricFunction,
                 output_transform: Callable[..., Tuple[torch.Tensor, torch.Tensor]] = _no_transform):
        super(LossFn, self).__init__(output_transform)
        self._loss_fn = loss_fn
        self._loss: Optional[torch.Tensor] = None

    def reset(self):
        self._loss = None

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self._loss = self._loss_fn(*output)
        if len(self._loss.shape) != 0:
            raise ValueError("loss_fn did not return the average loss.")

    def compute(self) -> torch.Tensor:
        if self._loss is None:
            raise NotComputableError(
                "Loss must have at least one example before it can be computed.")
        return self._loss


class LossForward(Metric):
    def __init__(self, output_transform=lambda x: x):
        super(LossForward, self).__init__(output_transform)
        self._loss = None

    def reset(self):
        self._loss = None

    def update(self, loss):
        self._loss = loss
        if self._loss is None:
            raise ValueError("Loss cannot be None.")

    def compute(self):
        if self._loss is None:
            raise NotComputableError(
                "Loss must have at least one example before it can be computed.")
        return self._loss.item()
