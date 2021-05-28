from datetime import datetime
from enum import auto
import os.path as osp
import subprocess
from typing import Optional

import numpy as np
import yaml

from ..data.datasets.build import DatasetType
from ..data.labels import LabelType
from ..models.build import ModelType
from ..utils.metrics import MetricType
from ..utils.path import expand_path
from .config import Config, ConfigEnum


class Mode(ConfigEnum):
    """Configuration mode, used to control the required parameters."""
    NEW = auto()
    CONTINUE = auto()
    INFERENCE = auto()
    TEST = auto()


def create_default_config(mode: Mode) -> Config:
    """Create default configuration with required and default parameters."""

    cfg = Config(allow_dynamic_params=True)

    # general
    cfg.define_param('extends', default=None)
    cfg.add_internal_param('mode', value=mode)

    cfg.define_param('base_dir', required=True)
    cfg.define_param('identifier', default=None)
    cfg.add_internal_param('experiment', value=None)
    cfg.define_param('checkpoint')
    cfg.define_param('device', default='cuda')

    # data
    training_data_required = (mode == Mode.NEW) or (mode == Mode.CONTINUE)

    data_grp = cfg.define_group('data')
    cfg.define_param('training', parent=data_grp, required=training_data_required)
    cfg.define_param('validation', parent=data_grp, required=False)
    cfg.define_param('dataset_type', parent=data_grp, required=True)
    cfg.define_param('sequential', parent=data_grp, default=False)

    # transforms
    transform_grp = cfg.define_group('transforms')
    cfg.define_param('on_validation', parent=transform_grp, default=False)
    cfg.define_param('nth_point', parent=transform_grp, default=1)
    cfg.define_param('nth_point_random', parent=transform_grp, default=False)
    cfg.define_param('min_range', parent=transform_grp, default=0.0)
    cfg.define_param('max_range', parent=transform_grp, default=np.inf)
    cfg.define_param('keep_probability', parent=transform_grp, default=1.0)
    cfg.define_param('max_points', parent=transform_grp, default=np.inf)
    cfg.define_param('fps', parent=transform_grp, default=np.inf)
    cfg.define_param('remove_transform', parent=transform_grp, default=False)

    pt_noise_grp = cfg.define_group('point_noise', parent=transform_grp)
    cfg.define_param('type', parent=pt_noise_grp, default='normal')
    cfg.define_param('scale', parent=pt_noise_grp, default=0.0)
    cfg.define_param('target_only', parent=pt_noise_grp, default=False)

    trans_noise_grp = cfg.define_group('translation_noise', parent=transform_grp)
    cfg.define_param('type', parent=trans_noise_grp, default='normal')
    cfg.define_param('scale', parent=trans_noise_grp, default=[0.0, 0.0, 0.0])

    rot_noise_grp = cfg.define_group('rotation_noise_deg', parent=transform_grp)
    cfg.define_param('type', parent=rot_noise_grp, default='normal')
    cfg.define_param('scale', parent=rot_noise_grp, default=[0.0, 0.0, 0.0])

    # data loader
    loader_grp = cfg.define_group('data_loader')
    cfg.define_param('parallel_loading', parent=loader_grp, default=False)
    cfg.define_param('num_workers', parent=loader_grp, default=0)
    cfg.define_param('batch_size', parent=loader_grp, default=1)
    cfg.define_param('buffer_size', parent=loader_grp, default=0)

    # model
    model_grp = cfg.define_group('model')
    cfg.define_param('weights', parent=model_grp)
    cfg.define_param('input_dim', parent=model_grp, default=3)
    cfg.define_param('point_dim', parent=model_grp, default=3)
    cfg.define_param('label_type', parent=model_grp, required=True)
    cfg.define_param('model_type', parent=model_grp, required=True)
    cfg.define_group('params', parent=model_grp)

    # metrics
    metrics_grp = cfg.define_group('metrics')
    cfg.define_param('loss', parent=metrics_grp, default=[])
    cfg.define_param('other', parent=metrics_grp, default=[])
    cfg.define_param('running_average_alpha', parent=metrics_grp, default=0.5)

    # solver
    optim_grp = cfg.define_group('optimizer')
    cfg.define_param('name', parent=optim_grp, default='Adam')
    cfg.define_param('max_epochs', parent=optim_grp)
    cfg.define_param('max_iterations', parent=optim_grp)
    cfg.define_param('base_lr', parent=optim_grp, default=0.0001)
    cfg.define_param('weight_decay', parent=optim_grp, default=0.0)
    cfg.define_param('bias_lr_factor', parent=optim_grp, default=2.0)
    cfg.define_param('weight_decay_bias', parent=optim_grp, default=0.0)
    cfg.define_param('accumulation_steps', parent=optim_grp, default=1)
    cfg.define_param('params', parent=optim_grp, default={})

    # scheduler
    scheduler_grp = cfg.define_group('scheduler')
    cfg.define_param('epoch', parent=scheduler_grp, default=None)
    cfg.define_param('iteration', parent=scheduler_grp, default=None)
    cfg.define_param('name', parent=scheduler_grp, default=None)
    cfg.define_param('on_iteration', parent=scheduler_grp, default=False)
    cfg.define_param('on_validation', parent=scheduler_grp, default=False)
    cfg.define_param('needs_metrics', parent=scheduler_grp, default=False)
    cfg.define_param('warmup_iterations', parent=scheduler_grp, default=0)
    cfg.define_param('warmup_multiplier', parent=scheduler_grp, default=1.0)
    cfg.define_param('params', parent=scheduler_grp, default={})

    # logging
    logging_grp = cfg.define_group('logging')
    cfg.define_param('add_graph', parent=logging_grp, default=False)
    cfg.define_param('summary_period', parent=logging_grp, default=5)
    cfg.define_param('log_period', parent=logging_grp, default=1000)
    cfg.define_param('checkpoint_period', parent=logging_grp, default=1000)
    cfg.define_param('checkpoint_n_saved', parent=logging_grp, default=10)
    cfg.define_param('validation_period', parent=logging_grp, default=5000)

    return cfg


def read_config(cfg: Config, f: str) -> None:
    """Read configuration data from file."""
    with open(f, 'r') as stream:
        d = yaml.load(stream, Loader=yaml.Loader)

    if 'extends' in d and d['extends'] is not None:
        extends = osp.realpath(osp.join(osp.dirname(f), d['extends']))
        if osp.realpath(f) != extends:
            read_config(cfg, extends)

    # store mode since it might be overwritten by read_dict
    mode = cfg.mode

    # read data
    cfg.read_dict(d)

    # reset mode and extends
    cfg.mode = mode
    cfg.extends = None


def finish_config(cfg: Config) -> None:
    """Finalize and freeze configuration."""
    # check if extends is left
    if cfg.extends is not None:
        raise RuntimeError("The extended config file was not loaded")

    # checkpoint and weights
    if cfg.mode == Mode.CONTINUE and cfg.checkpoint is None:
        raise RuntimeError("Please specify the checkpoint for continue")
    elif cfg.mode == Mode.INFERENCE and cfg.model.weights is None:
        raise RuntimeError("Please specify the model weights for inference")

    # full paths
    cfg.base_dir = expand_path(cfg.base_dir)
    cfg.checkpoint = expand_path(cfg.checkpoint)
    cfg.model.weights = expand_path(cfg.model.weights)
    cfg.data.training = expand_path(cfg.data.training)
    cfg.data.validation = expand_path(cfg.data.validation)

    # output directory
    if cfg.mode == Mode.NEW:
        cfg.experiment = datetime.now().strftime('%Y%m%d_%H%M%S')
        if cfg.identifier is not None:
            cfg.experiment += '_' + cfg.identifier
        cfg.output_dir = osp.join(cfg.base_dir, cfg.experiment)

    elif cfg.mode == Mode.CONTINUE:
        if cfg.experiment is not None:
            cfg.experiment += '_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            cfg.experiment = datetime.now().strftime('%Y%m%d_%H%M%S')
            if cfg.identifier is not None:
                cfg.experiment += '_' + cfg.identifier
        cfg.output_dir = osp.join(cfg.base_dir, cfg.experiment)

    else:
        cfg.output_dir = None

    # store git commit
    utils_path = osp.dirname(osp.realpath(__file__))
    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=utils_path).decode('utf-8').split('\n')[0]
    cfg.git_commit = commit_hash

    # check optimizer and scheduler settings
    if cfg.mode == Mode.NEW or cfg.mode == Mode.CONTINUE:
        if cfg.optimizer.max_epochs is None and cfg.optimizer.max_iterations is None:
            raise RuntimeError("Please define either max_epochs or max_iterations for the optimizer.")

        if cfg.scheduler.on_iteration and cfg.scheduler.on_validation:
            raise RuntimeError("Schedulers can either be executed on epoch, on iteration or on validation.")

    cfg.scheduler.on_epoch = not cfg.scheduler.on_iteration and not cfg.scheduler.on_validation

    # check loss
    if not isinstance(cfg.metrics.loss, list) or not isinstance(cfg.metrics.other, list):
        raise RuntimeError("Loss and other metrics have to be lists of metric configurations.")

    for i in range(len(cfg.metrics.loss)):
        cfg.metrics.loss[i]['type'] = MetricType.create(cfg.metrics.loss[i]['type'])
        if 'weights' not in cfg.metrics.loss[i]:
            cfg.metrics.loss[i]['weights'] = [1.0]
    for i in range(len(cfg.metrics.other)):
        cfg.metrics.other[i]['type'] = MetricType.create(cfg.metrics.other[i]['type'])

    # create types
    cfg.model.label_type = LabelType.create(cfg.model.label_type)
    cfg.model.model_type = ModelType.create(cfg.model.model_type)
    if cfg.mode != Mode.INFERENCE:
        cfg.data.dataset_type = DatasetType.create(cfg.data.dataset_type)

    # check dimensions
    if cfg.model.point_dim > cfg.model.input_dim:
        raise RuntimeError("Model input dimension must be equal or smaller than point dimension.")

    # freeze
    cfg.freeze()


def load_config(cfg_filename: str, mode: Mode, ckpt_filename: Optional[str] = None) -> Config:
    """Read, finalize and check configuration."""
    # create config
    config = create_default_config(mode=mode)

    # read config
    read_config(config, cfg_filename)
    if ckpt_filename is not None:
        config.checkpoint = ckpt_filename

    # finish and check config
    finish_config(config)
    if not config.is_valid():
        raise RuntimeError("Configuration is not valid, missing required parameters.")

    return config


def load_model_config(cfg_filename: str, weights_filename: str) -> Config:
    """Load configuration for model only."""
    config = create_default_config(mode=Mode.INFERENCE)
    config.model.read_file(cfg_filename)
    config.model.weights = weights_filename
    finish_config(config)
    return config.model
