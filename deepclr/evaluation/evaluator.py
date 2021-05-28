from __future__ import annotations
from collections import OrderedDict
import itertools
import os
import os.path as osp
from typing import Dict, List, Optional, OrderedDict as OrderedDictType, TypeVar

import matplotlib as mpl
import numpy as np

from .data import Sequence
from .metrics import SegmentMetrics, TransformationMetrics, MetricsContainer
from .plot import plot_error_over_time, plot_kitti_errors, plot_segment_error_bars, plot_sequence, plot_sequence_2d


_T = TypeVar('_T')

STEP_SIZE = 10  # every second
SEGMENT_LENGTHS = [100, 200, 300, 400, 500, 600, 700, 800]


def _step_errors(sequence: Sequence) -> List[TransformationMetrics]:
    """Calculate step-by-step errors for each transformation pair."""
    errors = [TransformationMetrics.calc(t_pred, t_gt, time)
              for t_pred, t_gt, time in zip(sequence.prediction.transforms, sequence.ground_truth.transforms,
                                            sequence.times)]
    return errors


def _segment_errors(sequence: Sequence, step_size: int = STEP_SIZE,
                    segment_lengths: Optional[List[int]] = None) -> List[SegmentMetrics]:
    """Calculate errors for larger segments as proposed by the KITTI Odometry evaluation."""
    assert len(sequence.prediction.poses) == len(sequence.ground_truth.poses)

    if segment_lengths is None:
        segment_lengths = SEGMENT_LENGTHS

    errors = []
    for first_frame in range(0, len(sequence.ground_truth.poses), step_size):
        for segment_length in segment_lengths:
            # last frame
            last_frame = sequence.ground_truth.get_frame_by_distance(first_frame, segment_length)
            if last_frame == -1:
                continue

            # segment length
            if segment_length == 0:
                segment_length = sequence.ground_truth.distances[last_frame] - \
                                 sequence.ground_truth.distances[first_frame]

            # speed
            num_frames = last_frame - first_frame + 1
            speed = segment_length / (0.1 * num_frames)

            # errors
            pose_delta_pred = np.dot(np.linalg.inv(sequence.prediction.poses[first_frame]),
                                     sequence.prediction.poses[last_frame])
            pose_delta_gt = np.dot(np.linalg.inv(sequence.ground_truth.poses[first_frame]),
                                   sequence.ground_truth.poses[last_frame])

            errors.append(SegmentMetrics.calc(pose_delta_pred, pose_delta_gt, first_frame=first_frame,
                                              segment_length=segment_length, speed=speed, normalize=True))

    return errors


def _merge_errors(errors: Dict[str, MetricsContainer]) -> MetricsContainer:
    merged_errors = list(itertools.chain.from_iterable(errors.values()))
    return MetricsContainer(merged_errors)


class Evaluator:
    """Evaluator class to store, process and visualize ground-truth and prediction data."""
    def __init__(self):
        self._sequences: OrderedDictType[str, Sequence] = OrderedDict()

        self._step_errors: Optional[OrderedDictType[str, MetricsContainer]] = None
        self._total_step_errors: Optional[MetricsContainer] = None
        self._segment_errors: Optional[OrderedDictType[str, MetricsContainer]] = None
        self._total_segment_errors: Optional[MetricsContainer] = None

    def reset(self):
        self._sequences.clear()
        self.reset_errors()

    def reset_errors(self):
        self._step_errors = None
        self._total_step_errors = None
        self._segment_errors = None
        self._total_segment_errors = None

    def add_transforms(self, name: str, stamp: float, pred: np.ndarray, gt: np.ndarray, time: float = 0) -> None:
        if name not in self._sequences:
            self._sequences[name] = Sequence()
        self._sequences[name].add_transforms(stamp, pred, gt, time)
        self.reset_errors()

    @classmethod
    def read(cls, path: str, filenames: Optional[List[str]] = None) -> Evaluator:
        if filenames is None:
            # get all files in path
            files = OrderedDict([(osp.splitext(f)[0], osp.join(path, f))
                                 for f in sorted(os.listdir(path))
                                 if osp.isfile(osp.join(path, f)) and f.endswith('.txt')])
        else:
            # create files from filenames
            files = OrderedDict([(osp.splitext(f)[0], osp.join(path, f))
                                 for f in filenames])

        # load files
        evaluator = Evaluator()
        for name, filename in files.items():
            evaluator._sequences[name] = Sequence.read(filename)
        return evaluator

    @classmethod
    def read_separate(cls, path_pred: str, path_gt: str, has_poses: bool, filenames: Optional[List[str]] = None) \
            -> Evaluator:
        if filenames is None:
            # get all files in path
            files_pred = OrderedDict([(osp.splitext(f)[0], f)
                                      for f in sorted(os.listdir(path_pred))
                                      if osp.isfile(osp.join(path_pred, f)) and f.endswith('.txt')])
        else:
            # create files from filenames
            files_pred = OrderedDict([(osp.splitext(f)[0], f)
                                      for f in filenames])

        evaluator = Evaluator()
        for name, filename in files_pred.items():
            # check if these files exist in ground truth path
            if not osp.isfile(osp.join(path_gt, filename)):
                raise RuntimeError(f"Could not find ground truth file for prediction '{filename}'")

            # load files
            evaluator._sequences[name] = Sequence.read_separate(osp.join(path_pred, filename),
                                                                osp.join(path_gt, filename), has_poses)

        return evaluator

    def write(self, path: str) -> None:
        for name, sequence in self._sequences.items():
            sequence.write(osp.join(path, f'{name}.txt'))

    def has_sequence(self, name: str) -> bool:
        return name in self._sequences

    def get_sequence(self, name: str) -> Sequence:
        return self._sequences[name]

    def get_sequences(self) -> OrderedDictType[str, Sequence]:
        return self._sequences

    def get_step_errors(self) -> OrderedDictType[str, MetricsContainer]:
        if self._step_errors is None:
            self._step_errors = OrderedDict([(name, MetricsContainer(_step_errors(sequence)))
                                             for name, sequence in self._sequences.items()])
        return self._step_errors

    def get_total_step_errors(self) -> MetricsContainer:
        if self._total_step_errors is None:
            self._total_step_errors = _merge_errors(self.get_step_errors())
        return self._total_step_errors

    def get_segment_errors(self) -> OrderedDictType[str, MetricsContainer]:
        if self._segment_errors is None:
            self._segment_errors = OrderedDict([(name, MetricsContainer(_segment_errors(sequence)))
                                                for name, sequence in self._sequences.items()])
        return self._segment_errors

    def get_total_segment_errors(self) -> MetricsContainer:
        if self._total_segment_errors is None:
            self._total_segment_errors = _merge_errors(self.get_segment_errors())
        return self._total_segment_errors

    def plot_error_over_time(self) -> OrderedDictType[str, mpl.figure.Figure]:
        return OrderedDict([(name, plot_error_over_time(step_errors))
                            for name, step_errors in self.get_step_errors().items()])

    def plot_kitti_errors(self) -> OrderedDictType[str, mpl.figure.Figure]:
        return OrderedDict([(name, plot_kitti_errors(segment_errors))
                            for name, segment_errors in self.get_segment_errors().items()])

    def plot_total_kitti_errors(self) -> mpl.figure.Figure:
        return plot_kitti_errors(self.get_total_segment_errors())

    def plot_segment_error_bars(self) -> mpl.figure.Figure:
        return plot_segment_error_bars(self.get_segment_errors())

    def plot_sequences(self) -> OrderedDictType[str, mpl.figure.Figure]:
        figures = OrderedDict()
        for name, sequence in self._sequences.items():
            fig = plot_sequence(sequence)
            fig.suptitle(f'{name}')
            figures[name] = fig
        return figures

    def plot_sequences_2d(self) -> OrderedDictType[str, mpl.figure.Figure]:
        figures = OrderedDict()
        for name, sequence in self._sequences.items():
            fig = plot_sequence_2d(sequence)
            fig.suptitle(f'{name}')
            figures[name] = fig
        return figures
