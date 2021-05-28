#!/usr/bin/env python3
import argparse

from deepclr.data.datasets.modelnet40 import ModelNet40PointClouds
from deepclr.utils.pcv import PointCloudVisualizer


def main():
    parser = argparse.ArgumentParser(description="Test ModelNet40.")
    parser.add_argument('filename', type=str, help="TXT file with list of all model files.")
    parser.add_argument('--shapes', default=None, type=str, nargs='*')
    args = parser.parse_args()

    # dataflow
    df = ModelNet40PointClouds(args.filename, args.shapes)

    # visualizer
    visualizer = PointCloudVisualizer()
    visualizer.set_window_size(640, 480)
    visualizer.set_background(0.5, 0.5, 0.5)
    visualizer.set_ground_plane(True, color=[0, 0, 0], alpha=0.5)

    # iterate data
    df.reset_state()
    for i, data in enumerate(df):
        cloud = data['cloud']
        if i % 10 == 0:
            print("Iteration {}/{}".format(i + 1, len(df)))
            visualizer.update_point_cloud('cloud', cloud[:, :3], color=[1, 0, 0], size=2)
            visualizer.spin_once(1000)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
