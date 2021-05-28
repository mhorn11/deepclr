#!/usr/bin/env python3
import os
import os.path as osp

import pykitti

from deepclr.utils.pcv import PointCloudVisualizer


SEQUENCE = '08'
INDEX = 50

COLOR1 = [31/255, 119/255, 180/255]
COLOR2 = [255/255, 127/255, 14/255]


def main():
    # path
    kitti_path = os.getenv('KITTI_PATH')
    if kitti_path is None:
        raise RuntimeError("Could not get KITTI path from environment variable KITTI_PATH.")

    # load
    original_path = osp.join(kitti_path, 'original')
    data = pykitti.odometry(original_path, SEQUENCE)

    # plot
    visualizer = PointCloudVisualizer()
    visualizer.set_background(1.0, 1.0, 1.0)

    template = data.get_velo(INDEX)
    source = data.get_velo(INDEX + 1)
    visualizer.update_point_cloud('template', template[:, :3], color=COLOR1)
    visualizer.add_point_cloud('source', source[:, :3], color=COLOR2)

    visualizer.spin()


if __name__ == '__main__':
    main()
