import os
import os.path as osp
import warnings

import numpy as np

from deepclr.data.datasets.modelnet40 import ModelNet40PointClouds


SHAPES = ['airplane']


def test_modelnet40():
    # path
    modelnet40_path = os.getenv('MODELNET40_PATH')
    if modelnet40_path is not None:
        modelnet40_filename = osp.join(modelnet40_path, 'original', 'modelnet40_test.txt')
    else:
        warnings.warn("Cannot test ModelNet40: MODELNET40_PATH not set")
        return

    # modelnet40
    modelnet40 = ModelNet40PointClouds(modelnet40_filename, shape_list=SHAPES, shuffle=False)
    assert len(modelnet40) == 100

    # first cloud
    for data in modelnet40:
        assert data['idx'] == 0
        assert isinstance(data['cloud'], np.ndarray)
        assert data['cloud'].shape[1] == 6
        break
