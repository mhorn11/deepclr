#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import warnings

import numpy as np
import pykitti

from deepclr.evaluation import Evaluator
from deepclr.data.datasets.kitti import velo2cam


SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
             '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']


def mat_to_vec(m: np.ndarray) -> np.ndarray:
    return m.reshape(1, 16)[0, :12]


def convert_poses(evaluator: Evaluator, kitti_base_path: str, sequence_name: str, output_dir: str) -> None:
    # load kitti calib
    kitti = pykitti.odometry(kitti_base_path, sequence_name)
    calib = kitti.calib.T_cam0_velo

    # iterate predicted poses
    sequence = evaluator.get_sequence(sequence_name)
    kitti_poses = [mat_to_vec(velo2cam(pose, calib))
                   for pose in sequence.prediction.poses]

    # save poses
    np.savetxt(osp.join(output_dir, f'{sequence_name}.txt'), np.array(kitti_poses))


def main():
    # parse inputs
    parser = argparse.ArgumentParser(description="Export predicted transformations as KITTI poses.")
    parser.add_argument('input_path', type=str, help="path with predicted transformations")
    args = parser.parse_args()

    # get kitti base path
    kitti_path = os.getenv('KITTI_PATH')
    if kitti_path is None:
        raise RuntimeError("Environment variable KITTI_PATH not defined.")
    kitti_base_path = osp.join(kitti_path, 'original')

    # load input files
    evaluator = Evaluator.read(args.input_path)

    # create output path
    output_dir = osp.join(args.input_path, 'kitti')
    os.makedirs(output_dir, exist_ok=True)

    # iterate sequences
    sequence_found = False
    for seq in SEQUENCES:
        # check sequence
        if not evaluator.has_sequence(seq):
            continue
        sequence_found = True

        # convert poses
        convert_poses(evaluator, kitti_base_path, seq, output_dir)

    # warning if no sequence was found
    if not sequence_found:
        warnings.warn("No sequence found in input directory.")


if __name__ == '__main__':
    main()
