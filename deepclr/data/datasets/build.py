from enum import auto
import os.path as osp
from typing import Any, Dict, List, Tuple, Union

from dataflow import ConcatData, DataFlow, ProxyDataFlow, RandomMixData
import numpy as np

from ...config.config import ConfigEnum
from .lmdb import LMDBSerializer, LMDBSequenceSerializer, LMDBSortedSerializer
from .utils import BatchDataQueue


class DatasetType(ConfigEnum):
    """Available dataset types."""
    GENERIC = auto()
    KITTI_ODOMETRY_VELODYNE = auto()
    MODELNET40 = auto()


class AttachDatasetName(ProxyDataFlow):
    """Attach dataset name to data points."""
    def __init__(self, ds: DataFlow, dataset: str):
        super().__init__(ds)
        self.dataset = dataset

    def __iter__(self):
        for dp in self.ds:
            dp['dataset'] = self.dataset
            yield dp


def _get_motion(m0: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """Calculate motion from two poses."""
    return np.linalg.inv(m0).dot(m1)


class MergePairSequence(ProxyDataFlow):
    """Merge list with two sequential clouds to a single data point."""
    def __init__(self, ds: DataFlow):
        super().__init__(ds)

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        for data in self.ds:
            assert len(data) == 2
            yield {'idx': [data[0]['idx'], data[1]['idx']],
                   'timestamps': [data[0]['timestamp'], data[1]['timestamp']],
                   'clouds': [data[0]['cloud'], data[1]['cloud']],
                   'transform': _get_motion(data[0]['pose'], data[1]['pose']),
                   'augmentations': [None, None]}


class DuplicateCloud(ProxyDataFlow):
    """Duplicate a cloud."""
    def __init__(self, ds: DataFlow):
        super().__init__(ds)

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        for data in self.ds:
            yield {'idx': [data['idx'], data['idx']],
                   'timestamps': [data['idx'], data['idx']],
                   'clouds': [data['cloud'], data['cloud'].copy()],
                   'transform': np.eye(4),
                   'augmentations': [None, None]}


class ToFloat32(ProxyDataFlow):
    """Convert all numpy arrays to float32."""
    def __init__(self, ds: DataFlow):
        super().__init__(ds)

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        for data in self.ds:
            yield self._to_float32(data)

    @staticmethod
    def _to_float32(x: Union[np.ndarray, Tuple[np.ndarray], List[np.ndarray], Dict[str, np.ndarray]]) \
            -> Any:
        if isinstance(x, (list, tuple)):
            return [ToFloat32._to_float32(v) for v in x]
        elif isinstance(x, dict):
            return {k: ToFloat32._to_float32(v) for k, v in x.items()}
        elif isinstance(x, np.ndarray):
            return x.astype(np.float32)
        else:
            return x


def create_input_dataflow(dataset_type: DatasetType, filename: str, shuffle: bool = False) -> DataFlow:
    """Create dataflow from a single dataset with a unified output structure:
    Dict[dataset: str, idx: List[int, int], timestamps: List[float, float], clouds: List[np.ndarray, np.ndarray],
    transform: np.ndarray, augmentations: List[np.ndarray, np.ndarray]}]"""

    if dataset_type == DatasetType.GENERIC:
        if shuffle:
            df = LMDBSerializer.load(filename, shuffle=True)
        else:
            df = LMDBSortedSerializer.load_sorted(filename)

    elif dataset_type == DatasetType.KITTI_ODOMETRY_VELODYNE:
        if shuffle:
            df = LMDBSequenceSerializer.load_sequence(filename, 2)
        else:
            df = LMDBSortedSerializer.load_sorted(filename)
            df = BatchDataQueue(df, 2, aggregate=False, use_list=True)
        df = MergePairSequence(df)
        df = AttachDatasetName(df, osp.splitext(osp.split(filename)[-1])[0])

    elif dataset_type == DatasetType.MODELNET40:
        if shuffle:
            df = LMDBSerializer.load(filename, shuffle=True)
        else:
            df = LMDBSortedSerializer.load_sorted(filename)
        df = DuplicateCloud(df)
        df = AttachDatasetName(df, osp.splitext(osp.split(filename)[-1])[0])

    else:
        raise NotImplementedError("DatasetType '{}' not implemented".format(dataset_type))

    # double to float
    df = ToFloat32(df)
    return df


def build_dataset(dataset_type: DatasetType, source: Union[str, List], shuffle: bool = False) -> DataFlow:
    """Create dataflow from multiple datasets with a unified output structure:
    Dict[dataset: str, idx: List[int, int], timestamps: List[float, float], clouds: List[np.ndarray, np.ndarray],
    transform: np.ndarray, augmentations: List[np.ndarray, np.ndarray]}]"""

    if isinstance(source, list):
        df = [create_input_dataflow(dataset_type, filename, shuffle=shuffle) for filename in source]
    else:
        df = create_input_dataflow(dataset_type, source, shuffle=shuffle)

    if isinstance(df, list):
        if shuffle:
            df = RandomMixData(df)
        else:
            df = ConcatData(df)

    return df
