from __future__ import annotations
from enum import auto
import os
import os.path as osp
import shutil
from typing import Type

from ..config.config import ConfigEnum, Config
from ..utils.checkpoint import load_model_state
from .base import BaseModel
from .deepclr import DeepCLR


class ModelType(ConfigEnum):
    DEEPCLR = auto()

    def get_class(self) -> Type[BaseModel]:
        if self == ModelType.DEEPCLR:
            return DeepCLR
        else:
            raise NotImplementedError("ModelType not implemented")


def build_model(model_cfg: Config) -> BaseModel:
    """Build model from config."""
    model_cls = model_cfg.model_type.get_class()
    args = {'input_dim': model_cfg.input_dim, 'point_dim': model_cfg.point_dim, 'label_type': model_cfg.label_type}
    model = model_cls(**args, **model_cfg.params)
    return model


def store_models_code(directory: str) -> None:
    """Store model source code."""
    models_directory = osp.dirname(osp.realpath(__file__))

    os.mkdir(directory)
    for filename in os.listdir(models_directory):
        source_file = osp.join(models_directory, filename)
        if osp.isfile(source_file):
            dest_file = osp.join(directory, filename)
            shutil.copyfile(source_file, dest_file)


def load_trained_model(model_cfg: Config) -> BaseModel:
    """Build model from config and load the state dictionary."""
    model = build_model(model_cfg)
    model_state_dict = load_model_state(model_cfg.weights)
    model.load_state_dict(model_state_dict)
    return model
