#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # noqa
import numpy as np

from deepclr.data.datasets.kitti import KittiOdometryVelodyneData
from deepclr.evaluation.plot import plot_path
from deepclr.utils.pcv import PointCloudVisualizer


def main():
    parser = argparse.ArgumentParser(description="Test KITTI Velodyne Odometry.")
    parser.add_argument('base_path', type=str)
    parser.add_argument('sequence', type=str)
    args = parser.parse_args()

    # dataflow
    df = KittiOdometryVelodyneData(args.base_path, args.sequence)

    # visualizer
    visualizer = PointCloudVisualizer()
    visualizer.set_window_size(640, 480)
    visualizer.set_background(0.5, 0.5, 0.5)
    visualizer.set_ground_plane(True, color=[0, 0, 0], alpha=0.5)

    # iterate data
    path_list = []
    df.reset_state()
    for i, data in enumerate(df):
        pose = data['pose']
        cloud = data['cloud']
        path_list.append(pose[:3, 3])

        if i % 100 == 0:
            print("Iteration {}/{}".format(i + 1, len(df)))
            visualizer.update_point_cloud('cloud', cloud[:, :3], color=[1, 0, 0], size=2)
            visualizer.spin_once(1000)

    path = np.array(path_list)

    # plot path
    plot_path(path)
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
