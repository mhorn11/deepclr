from datetime import timedelta

from dataflow import RNGDataFlow
import numpy as np
import pykitti


def _timedelta_to_us(td: timedelta) -> float:
    micros = td.days * 24 * 60 * 60 * 1000 * 1000 + \
             td.seconds * 1000 * 1000 + \
             td.microseconds
    return micros


def cam2velo(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Transform camera coordinate system pose to velodyne coordinate system pose using the calibration v."""
    v_inv = np.linalg.inv(v)
    return v_inv.dot(p).dot(v)


def velo2cam(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Transform velodyne coordinate system pose to camera coordinate system pose using the calibration v."""
    v_inv = np.linalg.inv(v)
    return np.dot(v, p).dot(v_inv)


class KittiOdometryVelodyneData(RNGDataFlow):
    """Read velodyne clouds and poses directly from KITTI odometry data."""
    def __init__(self, base_path: str, sequence: str, shuffle: bool = False):
        self.data = pykitti.odometry(base_path, sequence)
        self.calib = self.data.calib.T_cam0_velo
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        idxs = list(range(len(self.data)))
        if self.shuffle:
            self.rng.shuffle(idxs)

        for k in idxs:
            timestamp_us = _timedelta_to_us(self.data.timestamps[k])
            if len(self.data.poses) == 0:
                pose = np.eye(4)
            else:
                pose = cam2velo(self.data.poses[k], self.calib)
            cloud = self.data.get_velo(k)
            yield {'idx': k, 'timestamp': timestamp_us, 'pose': pose, 'cloud': cloud}


class KittiOdometryVelodyneSequenceData(RNGDataFlow):
    """Read consecutive velodyne sequences and poses directly from KITTI odometry data."""
    def __init__(self, base_path: str, sequence: str, seq_length: int, seq_step: int = 1, shuffle: bool = False):
        self.data = pykitti.odometry(base_path, sequence)
        self.calib = self.data.calib.T_cam0_velo

        self.seq_length = int(seq_length)
        assert self.seq_length > 0

        self.seq_step = int(seq_step)
        assert self.seq_step > 0

        self.shuffle = shuffle

        self.keys = range(len(self.data.timestamps))
        self.idxs = list(range(0, len(self.keys) - self.seq_length + 1, self.seq_step))

    def __len__(self):
        return len(self.idxs)

    def __iter__(self):
        idxs = self.idxs.copy()
        if self.shuffle:
            self.rng.shuffle(idxs)

        for start_idx in idxs:
            if len(self.data.poses) == 0:
                sequence = [{'idx': k,
                             'timestamp': _timedelta_to_us(self.data.timestamps[k]),
                             'pose': np.eye(4),
                             'cloud': self.data.get_velo(k)}
                            for k in range(start_idx, start_idx + self.seq_length)]
            else:
                sequence = [{'idx': k,
                             'timestamp': _timedelta_to_us(self.data.timestamps[k]),
                             'pose': cam2velo(self.data.poses[k], self.calib),
                             'cloud': self.data.get_velo(k)}
                            for k in range(start_idx, start_idx + self.seq_length)]
            yield sequence


class KittiSamplePairData(RNGDataFlow):
    """Read velodyne cloud and pose pairs as described by Lu et al. in 'DeepVCP: An End-to-End Deep Neural Network
    for Point Cloud Registration'"""
    def __init__(self, base_path: str, sequence: str, frame_interval: int, max_distance: float, shuffle: bool = False):
        self.data = pykitti.odometry(base_path, sequence)
        self.calib = self.data.calib.T_cam0_velo
        self.pairs = self._find_pairs(frame_interval, max_distance)
        self.shuffle = shuffle

    def _find_pairs(self, frame_interval, max_distance):
        pairs = []
        for i in range(0, len(self.data), frame_interval):
            for j in range(i + 1, len(self.data)):
                pose0 = cam2velo(self.data.poses[i], self.calib)
                pose1 = cam2velo(self.data.poses[j], self.calib)
                dist = np.linalg.norm(pose0[:3, 3] - pose1[:3, 3])
                if dist >= max_distance:
                    break
                pairs.append((i, j))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __iter__(self):
        idxs = list(range(len(self.pairs)))
        if self.shuffle:
            self.rng.shuffle(idxs)

        for k in idxs:
            idx0 = self.pairs[k][0]
            idx1 = self.pairs[k][1]

            timestamp1_us = _timedelta_to_us(self.data.timestamps[idx0])
            timestamp2_us = _timedelta_to_us(self.data.timestamps[idx1])
            if len(self.data.poses) == 0:
                pose0 = np.eye(4)
                pose1 = np.eye(4)
            else:
                pose0 = cam2velo(self.data.poses[idx0], self.calib)
                pose1 = cam2velo(self.data.poses[idx1], self.calib)
            cloud0 = self.data.get_velo(idx0)
            cloud1 = self.data.get_velo(idx1)

            yield [{'idx': idx0, 'timestamp': timestamp1_us, 'pose': pose0, 'cloud': cloud0},
                   {'idx': idx1, 'timestamp': timestamp2_us, 'pose': pose1, 'cloud': cloud1}]
