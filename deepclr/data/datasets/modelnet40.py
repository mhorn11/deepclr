from typing import List, Optional

import os.path as osp

from dataflow import RNGDataFlow
import numpy as np


class ModelNet40PointClouds(RNGDataFlow):
    """Read data points directly from ModelNet40 data, preprocessed by PointNet++ authors."""
    def __init__(self, filename: str, shape_list: Optional[List[str]] = None, shuffle: bool = False):
        names = [line.rstrip('\n') for line in open(filename)]
        directory = osp.dirname(filename)
        self.data = [osp.join(directory, name.rpartition('_')[0], f'{name}.txt') for name in names
                     if shape_list is None or name.rpartition('_')[0] in shape_list]
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        idxs = list(range(len(self.data)))
        if self.shuffle:
            self.rng.shuffle(idxs)

        for k in idxs:
            cloud = np.loadtxt(self.data[k], delimiter=',')
            yield {'idx': k, 'cloud': cloud}
