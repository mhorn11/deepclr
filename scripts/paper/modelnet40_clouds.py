#!/usr/bin/env python3
import os
import os.path as osp

import numpy as np

from deepclr.utils.pcv import PointCloudVisualizer
from deepclr.data.datasets.build import create_input_dataflow, DatasetType


INDEX = 1
NOISE = 0.04

COLOR1 = [31/255, 119/255, 180/255]
COLOR2 = [255/255, 127/255, 14/255]


def main():
    # path
    modelnet40_path = os.getenv('MODELNET40_PATH')
    if modelnet40_path is None:
        raise RuntimeError("Could not get ModelNet40 path from environment variable MODELNET40_PATH.")

    # load
    filename = osp.join(modelnet40_path, 'models', 'train.lmdb')
    df = create_input_dataflow(DatasetType.GENERIC, filename, shuffle=False)

    # plot
    visualizer = PointCloudVisualizer()
    visualizer.set_background(1.0, 1.0, 1.0)

    df.reset_state()
    for idx, data in enumerate(df):
        if idx != INDEX:
            continue
        template = data['cloud'][:, :3]
        source = template + np.random.normal(0, scale=NOISE, size=template.shape)
        visualizer.add_point_cloud('template', template, color=COLOR1, size=3)
        visualizer.add_point_cloud('source', source, color=COLOR2, size=3)
        visualizer.spin()
        break


if __name__ == '__main__':
    main()
