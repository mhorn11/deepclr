import copy
from typing import Dict, List, Optional, Union

import numpy as np
import scipy.spatial.distance
import transforms3d as t3d

from .utils import NoiseType, transform_point_cloud


_SampleType = Dict


class ApplyAugmentations(object):
    """Apply all augmentation transforms to the respective clouds."""
    def __init__(self, dim: int = 3):
        self.dim = dim
        if self.dim != 3:
            raise RuntimeError("Only three-dimensional transforms supported")

    def __call__(self, sample: _SampleType) -> _SampleType:
        for i, (cloud, augmentation) in enumerate(zip(sample['clouds'], sample['augmentations'])):
            if augmentation is not None:
                cloud = copy.copy(cloud)
                cloud[:, :self.dim] = transform_point_cloud(cloud[:, :self.dim], augmentation)
                sample['clouds'][i] = cloud
                sample['augmentations'][i] = None
        return sample


class FarthestPointSampling(object):
    """Perform farthest point sampling on all clouds."""
    def __init__(self, n: int, dim: int = 3):
        self.n = n
        self.dim = dim
        if self.dim != 3:
            raise RuntimeError("Only three-dimensional transforms supported")

    def __call__(self, sample: _SampleType) -> _SampleType:
        if 'cloud' in sample:
            sample['cloud'] = self._fps(sample['cloud'])
        else:
            for i, cloud in enumerate(sample['clouds']):
                sample['clouds'][i] = self._fps(cloud)
        return sample

    def _fps(self, cloud: np.ndarray) -> np.ndarray:
        if np.isinf(self.n) or cloud.shape[0] <= self.n:
            return cloud

        dist_mat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(cloud[:, :self.dim], 'euclidean'))
        perm = np.zeros(self.n, dtype=int)
        dist_vec = dist_mat[0, :]
        for i in range(1, self.n):
            idx = np.argmax(dist_vec)
            perm[i] = idx
            dist_vec = np.minimum(dist_vec, dist_mat[idx, :])

        return cloud[perm, :]


class PointNoise(object):
    """Add noise to point cloud samples."""
    def __init__(self, scale: float, noise_type: Optional[NoiseType] = None, target_only: bool = False, dim: int = 3):
        if noise_type is None:
            noise_type = NoiseType.NORMAL

        self.scale = scale
        self.noise_type = noise_type
        self.target_only = target_only
        self.dim = dim

    def __call__(self, sample: _SampleType) -> _SampleType:
        if self.scale <= 0.0:
            return sample

        if self.target_only:
            cloud = copy.copy(sample['clouds'][-1])
            cloud[:, :self.dim] += self.noise_type.get(self.scale, (int(cloud.shape[0]), self.dim))
            sample['clouds'][-1] = cloud
        else:
            for i, cloud in enumerate(sample['clouds']):
                cloud = copy.copy(cloud)
                cloud[:, :self.dim] += self.noise_type.get(self.scale, (cloud.shape[0], self.dim))
                sample['clouds'][i] = cloud

        return sample


class RangeSelection(object):
    """Remove points out of range."""
    def __init__(self, min_range: float, max_range: float, dim: int = 3):
        self.min_range = min_range
        self.max_range = max_range
        self.dim = dim
        if self.dim != 3:
            raise RuntimeError("Only three-dimensional transforms supported")

    def __call__(self, sample: _SampleType) -> _SampleType:
        sample['clouds'] = [self._range_selection(cloud) for cloud in sample['clouds']]
        return sample

    def _range_selection(self, cloud: np.ndarray) -> np.ndarray:
        if self.min_range == 0.0 and np.isinf(self.max_range):
            return cloud

        cloud_max = np.max(np.abs(cloud[:, :(self.dim - 1)]), axis=1)
        inliers = (cloud_max >= self.min_range) & (cloud_max <= self.max_range)

        return cloud[inliers, :]


class RandomErasing(object):
    """Randomly remove points."""
    def __init__(self, keep_probability: float, max_points: Union[int, float]):
        self.keep_probability = keep_probability
        self.max_points = max_points

    def __call__(self, sample: _SampleType) -> _SampleType:
        sample['clouds'] = [self._random_erasing(cloud) for cloud in sample['clouds']]
        return sample

    def _random_erasing(self, cloud: np.ndarray) -> np.ndarray:
        # keep probability
        if self.keep_probability < 1.0:
            keep_mask = np.random.rand(cloud.shape[0]) < self.keep_probability
            cloud = cloud[keep_mask, :]

        # max points
        if cloud.shape[0] > self.max_points:
            keep_idx = np.random.choice(np.arange(cloud.shape[0]), size=self.max_points, replace=False)
            cloud = cloud[keep_idx, :]

        return cloud


def _get_noise_type(x: Union[str, NoiseType]) -> NoiseType:
    if isinstance(x, str):
        return NoiseType[x.upper()]
    else:
        return x


_NoiseArgumentType = Union[str, NoiseType, List[str], List[NoiseType]]


class RandomTransform(object):
    """Randomly transform the second point cloud."""
    translation_noise_scale: List[float]
    rotation_noise_deg_scale: List[float]

    def __init__(self, translation_noise_scale: float, rotation_noise_deg_scale: float,
                 translation_noise_type: Optional[_NoiseArgumentType] = None,
                 rotation_noise_deg_type: Optional[_NoiseArgumentType] = None, dim: int = 3):
        if translation_noise_type is None:
            translation_noise_type = NoiseType.NORMAL
        if rotation_noise_deg_type is None:
            rotation_noise_deg_type = NoiseType.NORMAL

        if isinstance(translation_noise_scale, list):
            self.translation_noise_scale = translation_noise_scale
        else:
            self.translation_noise_scale = [translation_noise_scale for _ in range(dim)]

        if isinstance(rotation_noise_deg_scale, list):
            self.rotation_noise_deg_scale = rotation_noise_deg_scale
        else:
            self.rotation_noise_deg_scale = [rotation_noise_deg_scale for _ in range(dim)]

        if isinstance(translation_noise_type, list):
            self.translation_noise_type = [_get_noise_type(x) for x in translation_noise_type]
        else:
            self.translation_noise_type = [_get_noise_type(translation_noise_type) for _ in range(dim)]

        if isinstance(rotation_noise_deg_type, list):
            self.rotation_noise_deg_type = [_get_noise_type(x) for x in rotation_noise_deg_type]
        else:
            self.rotation_noise_deg_type = [_get_noise_type(rotation_noise_deg_type) for _ in range(dim)]

        self.dim = dim
        if self.dim != 3:
            raise RuntimeError("Only three-dimensional transforms supported")

        self.active = (np.sum([np.sum(np.abs(x)) for x in self.translation_noise_scale]) > 0.0) or \
                      (np.sum([np.sum(np.abs(x)) for x in self.rotation_noise_deg_scale]) > 0.0)

    def __call__(self, sample: _SampleType) -> _SampleType:
        if not self.active:
            return sample

        random_transform = self._get_random_transform()
        random_transform_cloud = np.linalg.inv(random_transform)

        if sample['augmentations'][-1] is None:
            sample['augmentations'][-1] = random_transform_cloud
        else:
            sample['augmentations'][-1] = np.dot(random_transform_cloud, sample['augmentations'][-1])
        sample['transform'] = np.dot(sample['transform'], random_transform)

        return sample

    def _get_random_transform(self) -> np.ndarray:
        # translation
        t = np.array([noise_type.get(noise_scale)
                      for noise_type, noise_scale in zip(self.translation_noise_type, self.translation_noise_scale)])

        # rotation euler angles
        rot_euler_deg = np.array([noise_type.get(noise_scale)
                                  for noise_type, noise_scale in zip(self.rotation_noise_deg_type,
                                                                     self.rotation_noise_deg_scale)])
        rot_euler = np.deg2rad(rot_euler_deg)
        r = t3d.euler.euler2mat(rot_euler[0], rot_euler[1], rot_euler[2], axes='sxyz')

        # affine
        z = np.ones(3)
        m = t3d.affines.compose(t, r, z)
        return m


class RemoveTransform(object):
    """Remove the ground truth transformation from the second point cloud."""
    def __init__(self, active: bool = True, dim: int = 3):
        self.active = active
        self.dim = dim
        if self.dim != 3:
            raise RuntimeError("Only three-dimensional transforms supported")

    def __call__(self, sample: _SampleType) -> _SampleType:
        if not self.active:
            return sample

        transform = sample['transform']

        if sample['augmentations'][-1] is None:
            sample['augmentations'][-1] = transform
        else:
            raise RuntimeError("RemoveTransform must be called before any other transform augmentation")

        sample['transform'] = np.eye(4)

        return sample


class SystematicErasing(object):
    """Systematically remove every nth point."""
    def __init__(self, nth: int, start: int = 0):
        self.nth = int(nth)
        self.start = int(start)
        assert self.nth >= 1
        assert -1 <= self.start < self.nth

    def __call__(self, sample: _SampleType) -> _SampleType:
        if 'cloud' in sample:
            sample['cloud'] = self._systematic_erasing(sample['cloud'])
        else:
            sample['clouds'] = [self._systematic_erasing(cloud) for cloud in sample['clouds']]
        return sample

    def _systematic_erasing(self, cloud: np.ndarray) -> np.ndarray:
        if self.nth == 1:
            return cloud

        if self.start == -1:
            start = int(np.random.uniform(0, self.nth))
        else:
            start = self.start

        return cloud[start::self.nth, :]


class TruncateDimension(object):
    """Truncate dimension of cloud points."""
    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    def __call__(self, sample: _SampleType) -> _SampleType:
        if 'cloud' in sample:
            sample['cloud'] = sample['cloud'][:, :self.input_dim]
        else:
            for i, cloud in enumerate(sample['clouds']):
                sample['clouds'][i] = cloud[:, :self.input_dim]
        return sample
