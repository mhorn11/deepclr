#!/usr/bin/env python3
import os
import os.path as osp

from dataflow import MapData
from dataflow.dataflow.serialize import LMDBSerializer
import torchvision

from deepclr.data import create_input_dataflow, DatasetType
from deepclr.data.transforms.transforms import ApplyAugmentations, RandomTransform
from deepclr.data.transforms.utils import NoiseType


SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
TRANSLATION = 1.0
ROTATION = 1.0


def convert_sequence(input_file: str, output_file: str) -> None:
    # input
    df = create_input_dataflow(DatasetType.GENERIC, input_file, shuffle=False)

    # transform
    transform = torchvision.transforms.Compose([
        RandomTransform(TRANSLATION, ROTATION,
                        translation_noise_type=NoiseType.UNIFORM,
                        rotation_noise_deg_type=NoiseType.UNIFORM),
        ApplyAugmentations()
    ])
    df = MapData(df, func=lambda x: transform(x))

    # output
    LMDBSerializer.save(df, output_file, write_frequency=5000)


def main():
    # get kitti paths
    kitti_path = os.getenv('KITTI_PATH')
    if kitti_path is None:
        raise RuntimeError("Environment variable KITTI_PATH not defined.")
    kitti_pairs_path = osp.join(kitti_path, 'pairs')
    kitti_pairs_test_path = osp.join(kitti_path, 'pairs_test')

    # create output directory
    os.makedirs(kitti_pairs_test_path, exist_ok=True)

    # iterate sequences
    for seq in SEQUENCES:
        print(f"Process sequence {seq}")
        input_file = osp.join(kitti_pairs_path, f'{seq}.lmdb')
        output_file = osp.join(kitti_pairs_test_path, f'{seq}.lmdb')
        convert_sequence(input_file, output_file)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
