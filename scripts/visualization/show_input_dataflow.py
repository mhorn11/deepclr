#!/usr/bin/env python3
import argparse

from deepclr.data import create_input_dataflow, DatasetType, transform_point_cloud
from deepclr.utils.pcv import PointCloudVisualizer


def main():
    parser = argparse.ArgumentParser(description="Test input dataflow.")
    parser.add_argument('dataset_type', type=str, help="input dataset type")
    parser.add_argument('filename', type=str, help="input filename")
    parser.add_argument('--shuffle', action='store_true', help="shuffle data randomly")
    args = parser.parse_args()

    # dataflow
    dataset_type = DatasetType[args.dataset_type.upper()]
    df = create_input_dataflow(dataset_type=dataset_type, filename=args.filename, shuffle=args.shuffle)

    # visualizer
    visualizer = PointCloudVisualizer()
    visualizer.set_window_size(640, 480)
    visualizer.set_background(0.5, 0.5, 0.5)
    visualizer.set_ground_plane(True, color=[0, 0, 0], alpha=0.5)

    df.reset_state()
    for dp in df:
        cloud0 = dp['clouds'][0][:, :3]
        cloud1 = dp['clouds'][1][:, :3]
        aug0 = dp['augmentations'][1]
        aug1 = dp['augmentations'][1]
        transform = dp['transform']

        if aug0 is not None:
            cloud0 = transform_point_cloud(cloud0, aug0)
        if aug1 is not None:
            cloud1 = transform_point_cloud(cloud1, aug1)
        cloud1t = transform_point_cloud(cloud1, transform)

        visualizer.update_point_cloud('cloud0', cloud0, color=[1, 1, 1], size=3)
        visualizer.update_point_cloud('cloud1', cloud1, color=[1, 0, 0], size=3)
        visualizer.update_point_cloud('cloud1t', cloud1t, color=[0, 1, 0], size=3)
        visualizer.spin_once(1000)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
