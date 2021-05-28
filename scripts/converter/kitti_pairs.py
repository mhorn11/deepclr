#!/usr/bin/env python3
import os
import os.path as osp

from dataflow import MapData
from dataflow.dataflow.serialize import LMDBSerializer
import torchvision

from deepclr.data.datasets.build import MergePairSequence, AttachDatasetName
from deepclr.data.datasets.kitti import KittiSamplePairData
from deepclr.data.transforms.transforms import ApplyAugmentations, RemoveTransform, SystematicErasing


SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
NTH = 2


def convert_sequence(base_path: str, sequence: str, output_file: str) -> None:
    # input
    df = KittiSamplePairData(base_path, sequence, frame_interval=30, max_distance=5.0, shuffle=False)
    df = MergePairSequence(df)
    df = AttachDatasetName(df, sequence)

    # transform
    transform = torchvision.transforms.Compose([
        RemoveTransform(),
        SystematicErasing(NTH),
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
    kitti_base_path = osp.join(kitti_path, 'original')
    kitti_pairs_path = osp.join(kitti_path, 'pairs')

    # create output directory
    os.makedirs(kitti_pairs_path, exist_ok=True)

    # iterate sequences
    for seq in SEQUENCES:
        print(f"Convert sequence {seq}")
        output_file = osp.join(kitti_pairs_path, f'{seq}.lmdb')
        convert_sequence(kitti_base_path, seq, output_file)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
