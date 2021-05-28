from collections import OrderedDict
import os
import os.path as osp
from typing import Dict, List, OrderedDict as OrderedDictType, Optional, TypedDict

from ignite.engine import Engine
import torch

from ..solver.schedulers import LRScheduler


class CheckpointData(TypedDict):
    epoch: int
    iteration: int
    model_state_dict: OrderedDictType[str, torch.Tensor]
    optimizer_state_dict: Dict
    scheduler_state_dict: Optional[Dict]


class Checkpointer:
    """Save checkpoints and remove old ones."""
    def __init__(self, directory: str, n_saved: int = 0, create_dir: bool = True):
        self.directory = directory
        self.n_saved = n_saved
        self.checkpoints: List[Dict[str, str]] = list()

        if create_dir:
            os.makedirs(directory, exist_ok=True)

    def save_checkpoint(self, engine: Engine, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        scheduler: Optional[LRScheduler] = None) -> None:
        # create checkpoint data
        data = create_checkpoint_data(engine, model, optimizer, scheduler)
        iteration = data['iteration']

        # store checkpoint and weights
        filenames = {'checkpoint': osp.join(self.directory, f'ckpt_{iteration}.tar'),
                     'weights': osp.join(self.directory, f'weights_{iteration}.tar')}
        torch.save(data, filenames['checkpoint'])
        torch.save(data['model_state_dict'], filenames['weights'])
        self.checkpoints.append(filenames)

        # update symlink
        self.update_symlinks(filenames)

        # remove old checkpoints
        if self.n_saved > 0:
            while len(self.checkpoints) > self.n_saved:
                for fname in self.checkpoints.pop(0).values():
                    os.remove(fname)

    def save_special_checkpoint(self, name: str, engine: Engine, model: torch.nn.Module,
                                optimizer: torch.optim.Optimizer, scheduler: Optional[LRScheduler] = None) -> None:
        # create checkpoint data
        data = create_checkpoint_data(engine, model, optimizer, scheduler)
        iteration = data['iteration']

        # store checkpoint and weights
        filenames = {'checkpoint': osp.join(self.directory, f'ckpt_{name}_{iteration}.tar'),
                     'weights': osp.join(self.directory, f'weights_{name}_{iteration}.tar')}
        torch.save(data, filenames['checkpoint'])
        torch.save(data['model_state_dict'], filenames['weights'])
        self.checkpoints.append(filenames)

        # update symlink
        self.update_symlinks(filenames)

    def update_symlinks(self, filenames: Dict[str, str]) -> None:
        for source_name, target_file in filenames.items():
            symlink = osp.join(self.directory, f'{source_name}.tar')
            if osp.isfile(symlink):
                os.remove(symlink)
            target = osp.relpath(target_file, self.directory)
            os.symlink(target, symlink)


def create_checkpoint_data(engine: Engine, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                           scheduler: Optional[LRScheduler] = None) -> CheckpointData:
    """Create checkpoint data structure for saving."""
    data: CheckpointData = {'epoch': engine.state.epoch,
                            'iteration': engine.state.iteration,
                            'model_state_dict': OrderedDict(model.state_dict()),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': None}

    if scheduler is not None:
        data['scheduler_state_dict'] = scheduler.state_dict()

    return data


def load_checkpoint(filename: str) -> CheckpointData:
    """Load checkpoint from file."""
    return torch.load(filename)


def load_model_state(filename: str) -> OrderedDictType[str, torch.Tensor]:
    """Load model state with weighs from file."""
    return torch.load(filename)
