from collections import deque
import copy
from typing import Callable, Optional

from dataflow.dataflow import BatchData, DataFlow, ProxyDataFlow


class BatchDataQueue(ProxyDataFlow):
    """Batch data points either as list or aggregate them."""
    def __init__(self, ds: DataFlow, batch_size: int, aggregate: bool = True, use_list: bool = False,
                 min_size: Optional[int] = None):
        super().__init__(ds)
        self.batch_size = int(batch_size)
        assert self.batch_size > 0
        self.aggregate = aggregate
        self.use_list = use_list

        self.min_size = self.batch_size
        if min_size is not None:
            self.min_size = min_size
        assert self.min_size > 0

    def __len__(self):
        return len(self.ds) - self.batch_size + 1

    def reset_state(self):
        super().reset_state()

    def __iter__(self):
        holder: deque = deque()
        for data in self.ds:
            holder.append(data)
            if len(holder) >= self.min_size:
                if self.aggregate:
                    yield BatchData.aggregate_batch(list(holder), self.use_list)
                else:
                    yield list(holder)
                if len(holder) == self.batch_size:
                    holder.popleft()


class MapDataList(ProxyDataFlow):
    """Apply function on each element of a data point list."""
    def __init__(self, ds: DataFlow, func: Callable):
        super().__init__(ds)
        self.func = func

    def __iter__(self):
        for dp in self.ds:
            ret = [self.func(copy.copy(el)) for el in dp]  # shallow copy the list elements
            yield ret
