from ..config.config import Config
from ..data.datasets.build import DatasetType
from ..utils.path import expand_path


def load_scenario(filename, with_method=False):
    """Load scenario config either with or without method data."""

    # define scenario config
    cfg = Config(allow_dynamic_params=True)
    cfg.define_param('name', required=True)
    cfg.define_param('dataset_type', required=True)
    cfg.define_param('sequential', required=True)
    cfg.define_param('data', required=True)

    method_grp = cfg.define_group('method')
    cfg.define_param('name', parent=method_grp, required=with_method)
    cfg.define_group('params', parent=method_grp)

    # read
    cfg.read_file(filename)

    # check and process
    if not cfg.is_valid():
        raise RuntimeError("Configuration is not valid, missing required parameters.")

    cfg.dataset_type = DatasetType.create(cfg.dataset_type)
    for data_name in cfg.data.keys():
        cfg.data[data_name] = expand_path(cfg.data[data_name])

    # freeze and return
    cfg.freeze()
    return cfg
