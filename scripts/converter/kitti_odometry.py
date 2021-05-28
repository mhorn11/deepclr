#!/usr/bin/env python3
import os
import os.path as osp

from dataflow import MapData
from dataflow.dataflow.serialize import LMDBSerializer

from deepclr.data.datasets.kitti import KittiOdometryVelodyneData
from deepclr.data.transforms.transforms import SystematicErasing


SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
             '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
NTH = 2


def convert_sequence(base_path: str, sequence: str, output_file: str) -> None:
    # input
    df = KittiOdometryVelodyneData(base_path, sequence, shuffle=False)

    # transform
    transform = SystematicErasing(NTH)
    df = MapData(df, func=lambda x: transform(x))

    # output
    LMDBSerializer.save(df, output_file, write_frequency=5000)


def main():
    # get kitti paths
    kitti_path = os.getenv('KITTI_PATH')
    if kitti_path is None:
        raise RuntimeError("Environment variable KITTI_PATH not defined.")
    kitti_base_path = osp.join(kitti_path, 'original')
    kitti_odometry_path = osp.join(kitti_path, 'odometry')

    # create output directory
    os.makedirs(kitti_odometry_path, exist_ok=True)

    # iterate sequences
    for seq in SEQUENCES:
        print(f"Convert sequence {seq}")
        output_file = osp.join(kitti_odometry_path, f'{seq}.lmdb')
        convert_sequence(kitti_base_path, seq, output_file)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
