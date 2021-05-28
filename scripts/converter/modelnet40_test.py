#!/usr/bin/env python3
import os
import os.path as osp

from dataflow import MapData
from dataflow.dataflow.serialize import LMDBSerializer
import torchvision

from deepclr.data import create_input_dataflow, DatasetType
from deepclr.data.transforms.transforms import ApplyAugmentations, PointNoise, RandomTransform
from deepclr.data.transforms.utils import NoiseType


NOISE_LEVELS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
TRANSLATION = 0.1
ROTATION = 5.0


def process_file(input_file: str, noise: float, output_file: str) -> None:
    # input
    df = create_input_dataflow(DatasetType.MODELNET40, input_file, shuffle=False)

    # transform
    transform = torchvision.transforms.Compose([
        RandomTransform(TRANSLATION, ROTATION,
                        translation_noise_type=NoiseType.UNIFORM,
                        rotation_noise_deg_type=NoiseType.UNIFORM),
        PointNoise(noise, noise_type=NoiseType.NORMAL, target_only=False),
        ApplyAugmentations()
    ])
    df = MapData(df, func=lambda x: transform(x))

    # output
    LMDBSerializer.save(df, output_file, write_frequency=5000)


def main():
    # get modelnet40 paths
    modelnet40_path = os.getenv('MODELNET40_PATH')
    if modelnet40_path is None:
        raise RuntimeError("Environment variable MODELNET40_PATH not defined.")
    modelnet40_models_path = osp.join(modelnet40_path, 'models')
    modelnet40_test_path = osp.join(modelnet40_path, 'test')

    # create output directory
    os.makedirs(modelnet40_test_path, exist_ok=True)

    # iterate noise levels
    for noise in NOISE_LEVELS:
        print(f"Process seen shapes with noise level '{noise:.2f}'")
        input_file_seen = osp.join(modelnet40_models_path, 'test_seen.lmdb')
        output_file_seen = osp.join(modelnet40_test_path, f'test_seen_{noise:.2f}.lmdb')
        process_file(input_file_seen, noise, output_file_seen)

        print(f"Process unseen shapes with noise level '{noise:.2f}'")
        input_file_unseen = osp.join(modelnet40_models_path, 'test_unseen.lmdb')
        output_file_unseen = osp.join(modelnet40_test_path, f'test_unseen_{noise:.2f}.lmdb')
        process_file(input_file_unseen, noise, output_file_unseen)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
