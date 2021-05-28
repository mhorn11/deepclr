from enum import Enum, auto
from typing import List, Optional, Tuple, Union

import numpy as np


class NoiseType(Enum):
    """Enumeration for random distributions, e.g., noise."""
    NORMAL = auto()
    UNIFORM = auto()
    UNIFORM_MINMAX = auto()

    def get(self, scale: Union[float, List[float], np.ndarray],
            size: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        if self == NoiseType.NORMAL:
            return np.random.normal(scale=scale, size=size)
        if self == NoiseType.UNIFORM:
            scale = np.array(scale)
            return np.random.uniform(low=-scale, high=scale, size=size)
        if self == NoiseType.UNIFORM_MINMAX:
            if isinstance(scale, (list, np.ndarray)):
                return np.random.uniform(low=scale[0], high=scale[1], size=size)
            else:
                raise TypeError("Invalid scale type for minmax noise.")
        raise NotImplementedError("NoiseType '{}' not implemented.".format(self))


def transform_point_cloud(cloud: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply (4,4) homogeneous transformation matrix on (3,n) point cloud."""
    cloud_affine = np.hstack((cloud, np.ones((cloud.shape[0], 1))))
    cloud_affine = np.dot(cloud_affine, np.transpose(transform))
    return cloud_affine[:, :-1]
