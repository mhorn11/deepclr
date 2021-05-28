import queue
import threading
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import torch

from dataflow import DataFlow, MapData, MultiProcessMapDataZMQ, MultiProcessRunnerZMQ, ProxyDataFlow

from ..config.config import Config
from .datasets.build import build_dataset
from .labels import LabelType
from .transforms.build import build_transform


class BatchDataNumpy(TypedDict):
    x: np.ndarray
    y: np.ndarray
    m: np.ndarray
    d: np.ndarray
    t: np.ndarray


class BatchDataTorch(TypedDict):
    x: torch.Tensor
    y: torch.Tensor
    m: torch.Tensor
    d: torch.Tensor
    t: torch.Tensor


class BatchRegistrationData(ProxyDataFlow):
    """Create batch from registration dataflow."""
    def __init__(self, ds: DataFlow, batch_size: int, label_type: LabelType, remainder: bool = False):
        super().__init__(ds)
        if not remainder:
            assert batch_size <= len(ds)
        self.batch_size = int(batch_size)
        assert self.batch_size > 0
        self.remainder = remainder
        self.label_type = label_type

    def __len__(self):
        ds_size = len(self.ds)
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)

    def __iter__(self):
        holder = []
        for data in self.ds:
            holder.append(data)
            if len(holder) == self.batch_size:
                yield self.aggregate_batch(holder)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield self.aggregate_batch(holder)

    def aggregate_batch(self, data_holder: List[Dict]) -> BatchDataNumpy:
        first_dp = data_holder[0]

        batch_dim = len(data_holder)
        cloud_count = len(first_dp['clouds'])
        cloud_shape0 = min([min([cloud.shape[0] for cloud in sample['clouds']]) for sample in data_holder])
        cloud_shape1 = first_dp['clouds'][0].shape[1]

        clouds_batch = np.empty((batch_dim * cloud_count, cloud_shape0, cloud_shape1), dtype=np.float32)
        labels_batch = np.empty((batch_dim, self.label_type.dim), dtype=np.float32)
        augmentations_batch = np.empty((batch_dim * cloud_count, 4, 4), dtype=np.float32)
        datasets_batch_list = [np.empty(0)] * batch_dim
        timestamps_batch = np.empty((batch_dim, 2), dtype=np.int64)

        for batch_idx, sample in enumerate(data_holder):
            # clouds
            for cloud_idx, (cloud, augmentation) in enumerate(zip(sample['clouds'], sample['augmentations'])):
                # reduce cloud for batch
                cloud_indices = np.random.choice(np.arange(cloud.shape[0]), cloud_shape0, replace=False)

                # store cloud
                clouds_batch[batch_idx + (cloud_idx * batch_dim), :, :] = cloud[cloud_indices, :]

                # store augmentation
                if augmentation is None:
                    augmentations_batch[batch_idx + (cloud_idx * batch_dim)] = np.eye(4)
                else:
                    augmentations_batch[batch_idx + (cloud_idx * batch_dim)] = augmentation

            # label
            labels_batch[batch_idx] = self.label_type.from_matrix(sample['transform'])
            datasets_batch_list[batch_idx] = sample['dataset']
            timestamps_batch[batch_idx] = sample['timestamps']

        datasets_batch = np.array(datasets_batch_list)

        output: BatchDataNumpy = {'x': clouds_batch, 'y': labels_batch, 'm': augmentations_batch,
                                  'd': datasets_batch, 't': timestamps_batch}
        return output


class ToTensor(ProxyDataFlow):
    """Create pytorch tensors from numpy arrays."""
    def __init__(self, ds: DataFlow):
        super().__init__(ds)

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        for data in self.ds:
            yield self._to_tensor(data)

    @staticmethod
    def _is_supported(x: np.ndarray) -> bool:
        return x.dtype in [np.bool_, np.float64, np.float32, np.float16,
                           np.int64, np.int32, np.int16, np.int8, np.uint8]

    @staticmethod
    def _to_tensor(x: Union[torch.Tensor, np.ndarray, Tuple[np.ndarray], List[np.ndarray], Dict[str, np.ndarray]]) \
            -> Any:
        if isinstance(x, (list, tuple)):
            return [ToTensor._to_tensor(v) for v in x]
        elif isinstance(x, dict):
            return {k: ToTensor._to_tensor(v) for k, v in x.items()}
        elif isinstance(x, np.ndarray) and ToTensor._is_supported(x):
            return torch.from_numpy(x)
        else:
            return x


class BufferQueue(ProxyDataFlow):
    """Buffer with separate worker thread for preloading data."""
    class WorkerThread(threading.Thread):
        def __init__(self, ds: DataFlow, buffer_size: int, *args: Any, **kwargs: Any):
            super(BufferQueue.WorkerThread, self).__init__(*args, **kwargs)
            self.ds = ds
            self.queue: queue.Queue = queue.Queue(buffer_size)

        def run(self) -> None:
            self.ds.reset_state()
            for pt in self.ds:
                self.queue.put(pt)
            self.queue.put(None)
            del self.ds

    def __init__(self, ds: DataFlow, buffer_size: int):
        super(BufferQueue, self).__init__(ds)
        self.buffer_size = buffer_size

    def reset_state(self):
        # reset_state is called by the worker
        pass

    def __iter__(self):
        self.worker = BufferQueue.WorkerThread(self.ds, self.buffer_size)
        self.worker.daemon = True
        self.worker.start()
        try:
            while True:
                pt = self.worker.queue.get()
                if pt is None:
                    break
                yield pt
        finally:
            del self.worker


def make_dataflow(cfg: Config, is_train: bool, source: Optional[Union[str, List]] = None,
                  batch_size: Optional[int] = None, tensor: bool = True) -> DataFlow:
    """Create complete dataflow with transforms, batching and buffering."""
    # dataset
    if source is None:
        source = cfg.data.training if is_train else cfg.data.validation
    df = build_dataset(cfg.data.dataset_type, source, shuffle=is_train)

    # transforms
    transform = build_transform(cfg, is_training=is_train)
    if is_train and not cfg.data_loader.parallel_loading and cfg.data_loader.num_workers > 0:
        df = MultiProcessMapDataZMQ(df, num_proc=cfg.data_loader.num_workers, map_func=lambda x: transform(x),
                                    buffer_size=cfg.data_loader.buffer_size, strict=True)
    else:
        df = MapData(df, func=lambda x: transform(x))

    # batch
    if batch_size is None:
        batch_size = cfg.data_loader.batch_size
    df = BatchRegistrationData(df, batch_size=batch_size, label_type=cfg.model.label_type, remainder=True)

    # parallel loading
    if is_train and cfg.data_loader.parallel_loading and cfg.data_loader.num_workers > 0:
        df = MultiProcessRunnerZMQ(df, num_proc=cfg.data_loader.num_workers, hwm=cfg.data_loader.buffer_size)

    # tensor
    if tensor:
        df = ToTensor(df)

    # queue
    if cfg.data_loader.buffer_size > 0:
        df = BufferQueue(df, buffer_size=cfg.data_loader.buffer_size)

    return df


class DataflowDataLoader:
    """Data loader with automatic reset."""
    _ds: Optional[DataFlow]

    def __init__(self, cfg: Config, is_train: bool, **kwargs: Any):
        self._cfg = cfg
        self._is_train = is_train
        self._kwargs = kwargs
        self._ds = None

    def _create_ds(self):
        if self._ds is None:
            self._ds = make_dataflow(self._cfg, self._is_train, **self._kwargs)
            self._ds.reset_state()

    def _stop_ds(self):
        if self._ds is not None:
            del self._ds
            self._ds = None

    def __len__(self):
        self._create_ds()
        if self._ds is not None:
            return len(self._ds)
        else:
            return 0

    def __iter__(self):
        self._create_ds()
        if self._ds is not None:
            for i, pt in enumerate(self._ds):
                yield pt
        self._stop_ds()


def make_data_loader(cfg: Config, is_train: bool, **kwargs: Any) -> DataflowDataLoader:
    return DataflowDataLoader(cfg, is_train, **kwargs)
