import os
import os.path as osp
import warnings

import numpy as np

from deepclr.data.datasets.kitti import KittiOdometryVelodyneData, KittiOdometryVelodyneSequenceData, \
    KittiSamplePairData


KITTI_PATH = os.getenv('KITTI_PATH')
KITTI_BASE_PATH = osp.join(KITTI_PATH, 'original') if KITTI_PATH is not None else None

SEQUENCE = '04'


def test_kitti_odometry_velodyne():
    if KITTI_BASE_PATH is None:
        warnings.warn('Cannot test KITTI: KITTI_PATH not set')
        return

    # kitti
    kitti = KittiOdometryVelodyneData(KITTI_BASE_PATH, SEQUENCE, shuffle=False)
    assert len(kitti) == 271

    # first cloud
    for data in kitti:
        assert data['idx'] == 0
        assert data['timestamp'] == 0.0
        assert isinstance(data['pose'], np.ndarray)
        assert data['pose'].shape == (4, 4)
        assert isinstance(data['cloud'], np.ndarray)
        assert data['cloud'].shape[1] == 4
        break


def test_kitti_odometry_velodyne_sequence():
    if KITTI_BASE_PATH is None:
        warnings.warn('Cannot test KITTI: KITTI_PATH not set')
        return

    # kitti
    kitti = KittiOdometryVelodyneSequenceData(KITTI_BASE_PATH, SEQUENCE, 3, shuffle=False)
    assert len(kitti) == 269

    # first cloud
    for data in kitti:
        assert len(data) == 3
        assert data[0]['idx'] == 0
        assert data[0]['timestamp'] == 0.0
        assert isinstance(data[0]['pose'], np.ndarray)
        assert data[0]['pose'].shape == (4, 4)
        assert isinstance(data[0]['cloud'], np.ndarray)
        assert data[0]['cloud'].shape[1] == 4
        break


def test_kitti_sample_pairs():
    if KITTI_BASE_PATH is None:
        warnings.warn('Cannot test KITTI: KITTI_PATH not set')
        return

    # kitti
    kitti = KittiSamplePairData(KITTI_BASE_PATH, SEQUENCE, frame_interval=30, max_distance=5.0, shuffle=False)

    # first cloud
    for data in kitti:
        assert len(data) == 2
        assert data[0]['idx'] == 0
        assert data[0]['timestamp'] == 0.0
        assert isinstance(data[0]['pose'], np.ndarray)
        assert data[0]['pose'].shape == (4, 4)
        assert isinstance(data[0]['cloud'], np.ndarray)
        assert data[0]['cloud'].shape[1] == 4
        break
