#!/usr/bin/env python3
import argparse
import os
import os.path as osp
from typing import Dict, Optional
import warnings

import numpy as np
import pandas as pd

from deepclr.config import Config
from deepclr.evaluation import Evaluator, load_scenario


SCENARIO_NAME = 'kitti_pairs'


def load_scenario_from_dir(directory: str) -> Optional[Config]:
    # load scenario
    sceneario_file = osp.join(directory, 'scenario.yaml')
    if not osp.isfile(sceneario_file):
        return None
    try:
        return load_scenario(sceneario_file, with_method=True)
    except RuntimeError:
        warnings.warn(f"Scenario invalid: '{sceneario_file}'")
        return None


def evaluate(path: str, scenario: Config) -> Dict:
    # load
    filenames = [f'{k}.txt' for k in scenario.data.keys()]
    evaluator = Evaluator.read(path, filenames)

    # process
    metrics = evaluator.get_total_step_errors()

    # convert
    data = {
        'Rot. Error Mean [deg]': np.rad2deg(metrics.mean.rotation.chordal),
        'Rot. Error Max [deg]': np.rad2deg(metrics.max.rotation.chordal),
        'Tran. Error Mean [m]': metrics.mean.translation.kitti,
        'Tran. Error Max [m]': metrics.max.translation.kitti,
        'Time [ms]': metrics.mean.time
    }
    return data


def main():
    # parse inputs
    parser = argparse.ArgumentParser(description="Print table for artificial KITTI evaluation.")
    parser.add_argument('path', type=str, help="base directory of inference output")
    args = parser.parse_args()

    base_path = args.path

    # search scenarios
    data = {}
    for dirname in sorted(os.listdir(base_path)):
        # check directory
        directory = osp.join(base_path, dirname)
        if not osp.isdir(directory):
            continue

        # load and check scenario
        scenario = load_scenario_from_dir(directory)
        if scenario is not None and scenario.name == SCENARIO_NAME:
            data[scenario.method.name] = evaluate(directory, scenario)

    # warning if no scenario was found
    if len(data) == 0:
        warnings.warn("Could not find scenario.")
    else:
        df = pd.DataFrame(data)
        print("== DeepCLR Results on KITTI Data with Artificial Transformations ==")
        print(df.transpose())


if __name__ == '__main__':
    main()
