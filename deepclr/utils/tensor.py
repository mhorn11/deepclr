from typing import Optional, Union

from ignite._utils import convert_tensor
import torch


def prepare_tensor(x: torch.Tensor, device: Optional[Union[str, torch.device]] = None, non_blocking: bool = False)\
        -> torch.Tensor:
    """Prepare tensor: pass to a device with options."""
    return convert_tensor(x, device=device, non_blocking=non_blocking)
