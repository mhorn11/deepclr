from __future__ import annotations
from typing import List

import numpy as np


def _vec_to_mat(v: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :] = v.reshape(3, 4)
    return m


def _mat_to_vec(m: np.ndarray) -> np.ndarray:
    return m.reshape(1, 16)[0, :12]


class Motion:
    """Store poses and the respective transforms and traveled distances together."""
    def __init__(self):
        self.transforms: List[np.ndarray] = list()
        self.poses: List[np.ndarray] = list()
        self.distances = list()

    def add_transform(self, m: np.ndarray) -> None:
        # transform
        self.transforms.append(m)

        # pose
        if len(self.poses) == 0:
            self.poses.append(np.eye(4))
            self.distances.append(0)
        self.poses.append(np.dot(self.poses[-1], m))

        # distance
        dist = np.linalg.norm(m[:3, 3], ord=2)
        self.distances.append(self.distances[-1] + dist)

    def add_pose(self, m: np.ndarray) -> None:
        # poses
        self.poses.append(m)

        if len(self.poses) > 1:
            # transform
            transform = np.dot(np.linalg.inv(self.poses[-2]), self.poses[-1])
            self.transforms.append(transform)

            # distance
            dist = np.linalg.norm(transform[:3, 3], ord=2)
            self.distances.append(self.distances[-1] + dist)
        else:
            self.distances.append(0)

    def get_path(self) -> np.ndarray:
        return np.array([p[:3, 3] for p in self.poses])

    def get_frame_by_distance(self, first_frame: int, distance: float) -> int:
        for i in range(first_frame, len(self.distances)):
            if self.distances[i] > self.distances[first_frame] + distance:
                return i
        return -1

    @classmethod
    def read(cls, filename: str, has_poses: bool) -> Motion:
        motion = cls()
        data = np.loadtxt(filename)
        for row in range(data.shape[0]):
            m = _vec_to_mat(data[row, :12])
            if has_poses:
                motion.add_transform(m)
            else:
                motion.add_pose(m)
        return motion

    def write(self, filename: str, use_poses: bool) -> None:
        data = list()
        export = self.poses if use_poses else self.transforms
        for m in export:
            data.append(_mat_to_vec(m))
        np.savetxt(filename, np.array(data))


class Sequence:
    """Store predicted and ground-truth motion along with data timestamps and inference times."""
    def __init__(self):
        self.prediction = Motion()
        self.ground_truth = Motion()
        self.stamps = list()
        self.times = list()

    def add_transforms(self, stamp: float, pred: np.ndarray, gt: np.ndarray, time: float = 0) -> None:
        self.stamps.append(stamp)
        self.prediction.add_transform(pred)
        self.ground_truth.add_transform(gt)
        self.times.append(time)

    def add_poses(self, stamp: float, pred: np.ndarray, gt: np.ndarray, time: float = 0) -> None:
        self.stamps.append(stamp)
        self.prediction.add_pose(pred)
        self.ground_truth.add_pose(gt)
        self.times.append(time)

    @classmethod
    def read(cls, filename: str) -> Sequence:
        sequence = cls()
        data = np.loadtxt(filename)
        for row in range(data.shape[0]):
            stamp = data[row, 0]
            pred = _vec_to_mat(data[row, 1:13])
            gt = _vec_to_mat(data[row, 13:25])
            time = data[row, 25]
            sequence.add_transforms(stamp, pred, gt, time)
        return sequence

    @classmethod
    def read_separate(cls, filename_pred: str, filename_gt: str, has_poses: bool) -> Sequence:
        sequence = cls()
        sequence.prediction = Motion.read(filename_pred, has_poses)
        sequence.ground_truth = Motion.read(filename_gt, has_poses)

        # check size
        size = len(sequence.prediction.transforms)
        if len(sequence.ground_truth.transforms) != size:
            raise RuntimeError("Sizes of prediction and ground truth files do not match.")

        # create dummy stamps and times
        sequence.stamps = np.arange(size).tolist()
        sequence.times = np.zeros(size).tolist()

        return sequence

    def write(self, filename: str) -> None:
        data = list()
        for stamp, pred, gt, time in zip(self.stamps, self.prediction.transforms,
                                         self.ground_truth.transforms, self.times):
            row = np.concatenate(([stamp], _mat_to_vec(pred), _mat_to_vec(gt), [time]))
            data.append(row)
        np.savetxt(filename, np.array(data))
