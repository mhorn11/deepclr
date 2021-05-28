import argparse
import sys

from kitti_devkit_ import eval


if __name__ == '__main__':
    # parse inputs
    parser = argparse.ArgumentParser(prog='kitti_devkit', description="KITTI Devkit Evaluation")
    parser.add_argument('gt_dir', type=str, help="Directory with ground-truth poses")
    parser.add_argument('pred_dir', type=str, help="Directory with predicted poses")
    args = parser.parse_args()

    # run eval
    sys.exit(eval(args.gt_dir, args.pred_dir))
