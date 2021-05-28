#!/usr/bin/env python3
import argparse

from deepclr.config import Mode, load_config
from deepclr.data import make_data_loader, transform_point_cloud
from deepclr.utils.pcv import PointCloudVisualizer


def main():
    parser = argparse.ArgumentParser(description="Test data loader.")
    parser.add_argument('config', type=str, help="training configuration (*.yaml)")
    parser.add_argument('--training', action='store_true', help="training or validation mode")
    args = parser.parse_args()

    # cfg
    cfg = load_config(args.config, Mode.TEST)

    # data loader
    loader = make_data_loader(cfg, is_train=args.training)

    # visualizer
    visualizer = PointCloudVisualizer()
    visualizer.set_window_size(640, 480)
    visualizer.set_background(0.5, 0.5, 0.5)
    visualizer.set_ground_plane(True, color=[0, 0, 0], alpha=0.5)

    for sample in loader:
        print("Next sample...")
        x, y, m = sample['x'], sample['y'], sample['m']
        dim = int(x.shape[0] / 2)
        for i in range(dim):
            cloud0 = x[i, :, :3]
            cloud1 = x[i + dim, :, :3]
            aug0 = m[i]
            aug1 = m[i + dim]
            transform = cfg.model.label_type.to_matrix(y[i])

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
