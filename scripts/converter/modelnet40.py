#!/usr/bin/env python3
import os
import os.path as osp
from typing import List

from dataflow import MapData
from dataflow.dataflow.serialize import LMDBSerializer

from deepclr.data.datasets.modelnet40 import ModelNet40PointClouds
from deepclr.data.transforms.transforms import FarthestPointSampling


SHAPES_SEEN = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup',
               'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp']
SHAPES_UNSEEN = ['laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood',
                 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
FPS = 2048


def process_file(input_file: str, shapes: List[str], output_file: str) -> None:
    # input
    df = ModelNet40PointClouds(input_file, shapes, shuffle=False)

    # transform
    transform = FarthestPointSampling(FPS)
    df = MapData(df, func=lambda x: transform(x))

    # output
    LMDBSerializer.save(df, output_file, write_frequency=5000)


def main():
    # get modelnet40 paths
    modelnet40_path = os.getenv('MODELNET40_PATH')
    if modelnet40_path is None:
        raise RuntimeError("Environment variable MODELNET40_PATH not defined.")
    modelnet40_original_path = osp.join(modelnet40_path, 'original')
    modelnet40_models_path = osp.join(modelnet40_path, 'models')

    # create output directory
    os.makedirs(modelnet40_models_path, exist_ok=True)

    # iterate files
    processing = [('modelnet40_train.txt', SHAPES_SEEN, 'train.lmdb'),
                  ('modelnet40_test.txt', SHAPES_SEEN, 'test_seen.lmdb'),
                  ('modelnet40_test.txt', SHAPES_UNSEEN, 'test_unseen.lmdb')]

    for input_filename, shapes, output_filename in processing:
        print(f"Create '{output_filename}'")
        input_file = osp.join(modelnet40_original_path, input_filename)
        output_file = osp.join(modelnet40_models_path, output_filename)
        process_file(input_file, shapes, output_file)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
