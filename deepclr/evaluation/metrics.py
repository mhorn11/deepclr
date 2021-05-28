from __future__ import annotations
from typing import Any, Callable, Dict, List, Sequence, Tuple, TypeVar, Union

import numpy as np
import transforms3d as t3d


def _translation_error_kitti(diff: np.typing.ArrayLike) -> Tuple[float, np.ndarray]:
    """Calculate translation error as proposed by the KITTI Odometry evaluation."""
    assert isinstance(diff, np.ndarray)
    err = np.linalg.norm(diff[:3, 3], ord=2)
    err_vec = diff[:3, 3]
    return err, err_vec


def translation_error_kitti(m1: np.ndarray, m2: np.ndarray) -> Tuple[float, np.ndarray]:
    """Calculate translation error as proposed by the KITTI Odometry evaluation."""
    err1, err_vec1 = _translation_error_kitti(m1.dot(np.linalg.inv(m2)))
    err2, err_vec2 = _translation_error_kitti(m2.dot(np.linalg.inv(m1)))
    return (err1, err_vec1) if err1 < err2 else (err2, err_vec2)


def translation_error_rmse(m1: np.ndarray, m2: np.ndarray) -> float:
    """Calculate RMSE translations error (this is NOT our recommended approach!)."""
    diff = m1[:3, 3] - m2[:3, 3]
    return np.sqrt(np.sum(np.square(diff)) / 3.0)


def _rotation_error_kitti(diff: np.typing.ArrayLike) -> Tuple[float, np.ndarray]:
    """Calculate rotation error as proposed by the KITTI Odometry evaluation."""
    assert isinstance(diff, np.ndarray)
    a = diff[0, 0]
    b = diff[1, 1]
    c = diff[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    err = np.arccos(max(min(d, 1.0), -1.0))

    _, R, _, _ = t3d.affines.decompose(diff)
    roll, pitch, yaw = t3d.euler.mat2euler(R, axes='sxyz')
    err_vec = np.array([roll, pitch, yaw])

    return err, err_vec


def rotation_error_kitti(m1: np.ndarray, m2: np.ndarray) -> Tuple[float, np.ndarray]:
    """Calculate rotation error as proposed by the KITTI Odometry evaluation."""
    err1, err_vec1 = _rotation_error_kitti(m1.dot(np.linalg.inv(m2)))
    err2, err_vec2 = _rotation_error_kitti(m2.dot(np.linalg.inv(m1)))
    return (err1, err_vec1) if err1 < err2 else (err2, err_vec2)


def rotation_error_rsme(m1: np.ndarray, m2: np.ndarray) -> float:
    """Calculate RMSE rotation error (this is NOT our recommended approach!)."""
    roll1, pitch1, yaw1 = t3d.euler.mat2euler(m1[:3, :3], axes='sxyz')
    roll2, pitch2, yaw2 = t3d.euler.mat2euler(m2[:3, :3], axes='sxyz')
    return np.sqrt(((roll1 - roll2) ** 2 + (pitch1 - pitch2) ** 2 + (yaw1 - yaw2) ** 2) / 3.0)


def rotation_error_chordal(m1: np.ndarray, m2: np.ndarray) -> float:
    """Calculate Chordal rotation error."""
    rot_diff = m1[:3, :3] - m2[:3, :3]
    rot_diff_norm = np.linalg.norm(rot_diff, ord='fro') / np.sqrt(8)
    err = 2 * np.arcsin(rot_diff_norm / np.sqrt(8))
    return err


class TranslationError:
    """Calculate and store multiple translation error metrics."""
    def __init__(self, kitti: float, rmse: float, vec: np.ndarray):
        self.kitti = kitti
        self.rmse = rmse
        self.vec = vec

    @classmethod
    def calc(cls, m1: np.ndarray, m2: np.ndarray) -> TranslationError:
        kitti, vec = translation_error_kitti(m1, m2)
        rmse = translation_error_rmse(m1, m2)
        return cls(kitti, rmse, vec)

    def divide(self, x):
        self.kitti = self.kitti / x
        self.vec = self.vec / x
        self.rmse = self.kitti / x

    @staticmethod
    def metrics() -> List[str]:
        return ['kitti', 'rmse', 'vec']


class RotationError:
    """Calculate and store multiple rotation error metrics."""
    def __init__(self, kitti: float, rmse: float, chordal: float, vec: np.ndarray):
        self.kitti = kitti
        self.rmse = rmse
        self.chordal = chordal
        self.vec = vec

    @classmethod
    def calc(cls, m1: np.ndarray, m2: np.ndarray) -> RotationError:
        kitti, vec = rotation_error_kitti(m1, m2)
        rmse = rotation_error_rsme(m1, m2)
        chordal = rotation_error_chordal(m1, m2)
        return cls(kitti, rmse, chordal, vec)

    def divide(self, x):
        self.kitti = self.kitti / x
        self.rmse = self.kitti / x
        self.chordal = self.kitti / x
        self.vec = self.vec / x

    @staticmethod
    def metrics() -> List[str]:
        return ['kitti', 'rmse', 'chordal', 'vec']


class TransformationMetrics:
    """Calculate and store multiple translation and rotation error metrics along with the inference time."""
    def __init__(self, translation: TranslationError, rotation: RotationError, time: float):
        self.translation = translation
        self.rotation = rotation
        self.time = time

    @classmethod
    def calc(cls, pred: np.ndarray, gt: np.ndarray, time: float) -> TransformationMetrics:
        translation = TranslationError.calc(pred, gt)
        rotation = RotationError.calc(pred, gt)
        return cls(translation, rotation, time)


class SegmentMetrics:
    """Calculate and store multiple segment-based metrics as proposed by the KITTI Odometry evaluation."""
    def __init__(self, translation: TranslationError, rotation: RotationError, first_frame: int,
                 segment_length: int, speed: float):
        self.translation = translation
        self.rotation = rotation
        self.first_frame = first_frame
        self.segment_length = segment_length
        self.speed = speed

    @classmethod
    def calc(cls, pred: np.ndarray, gt: np.ndarray, first_frame: int, segment_length: int, speed: float,
             normalize: bool) -> SegmentMetrics:
        translation = TranslationError.calc(pred, gt)
        rotation = RotationError.calc(pred, gt)
        if normalize and segment_length > 0:
            translation.divide(segment_length)
            rotation.divide(segment_length)
        return cls(translation, rotation, first_frame, segment_length, speed)


def _apply_function(func: Callable, data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: func(v) for k, v in data.items()}


_ErrorType = TypeVar('_ErrorType', TransformationMetrics, SegmentMetrics)


class MetricsContainer:
    """Store sequence of metrics and inference times along with various accumulations."""
    def __init__(self, data: Sequence[Union[TransformationMetrics, SegmentMetrics]]):
        # store data
        self.data = data

        # create arrays for each metric
        trans_arrs = {metric: np.array([getattr(x.translation, metric) for x in data])
                      for metric in TranslationError.metrics()}
        rot_arrs = {metric: np.array([getattr(x.rotation, metric) for x in data])
                    for metric in RotationError.metrics()}
        time_arr = np.array([x.time if isinstance(x, TransformationMetrics) else 0.0
                             for x in data])

        # get stats
        def min_func(x): return np.min(x, axis=0)
        def max_func(x): return np.max(x, axis=0)
        def mean_func(x): return np.mean(x, axis=0)
        def median_func(x): return np.median(x, axis=0)
        def std_func(x): return np.std(x, axis=0)

        self.min = TransformationMetrics(TranslationError(**_apply_function(min_func, trans_arrs)),
                                         RotationError(**_apply_function(min_func, rot_arrs)),
                                         min_func(time_arr))
        self.max = TransformationMetrics(TranslationError(**_apply_function(max_func, trans_arrs)),
                                         RotationError(**_apply_function(max_func, rot_arrs)),
                                         max_func(time_arr))
        self.mean = TransformationMetrics(TranslationError(**_apply_function(mean_func, trans_arrs)),
                                          RotationError(**_apply_function(mean_func, rot_arrs)),
                                          mean_func(time_arr))
        self.median = TransformationMetrics(TranslationError(**_apply_function(median_func, trans_arrs)),
                                            RotationError(**_apply_function(median_func, rot_arrs)),
                                            median_func(time_arr))
        self.std = TransformationMetrics(TranslationError(**_apply_function(std_func, trans_arrs)),
                                         RotationError(**_apply_function(std_func, rot_arrs)),
                                         std_func(time_arr))

    def __getitem__(self, i):
        return self.data[i]

    def __iter__(self):
        for x in self.data:
            yield x

    def __len__(self):
        return len(self.data)
