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


SCENARIO_NAME = 'modelnet40_unseen'
DATASET_NAME = 'test_unseen_0.04'


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


def evaluate(path: str, scenario: Config) -> Optional[Dict]:
    # load
    filenames = [f'{k}.txt' for k in scenario.data.keys()]
    evaluator = Evaluator.read(path, filenames)

    # process
    step_errors = evaluator.get_step_errors()
    if DATASET_NAME not in step_errors:
        warnings.warn('Dataset not found in scenario.')
        return None

    # convert
    metrics = step_errors[DATASET_NAME]
    data = {
        'Rot. Error Mean [deg]': np.rad2deg(metrics.mean.rotation.chordal),
        'Rot. Error Std [deg]': np.rad2deg(metrics.std.rotation.chordal),
        'Tran. Error Mean [m]': metrics.mean.translation.kitti,
        'Tran. Error Std [m]': metrics.std.translation.kitti,
        'Time [ms]': metrics.mean.time
    }
    return data


def main():
    # parse inputs
    parser = argparse.ArgumentParser(description="Print table for ModelNet40 evaluation.")
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
            scenario_data = evaluate(directory, scenario)
            if scenario_data is not None:
                data[scenario.method.name] = scenario_data

    # warning if no scenario was found
    if len(data) == 0:
        warnings.warn("Could not find scenario.")
    else:
        df = pd.DataFrame(data)
        print(f"== DeepCLR Results on Unseen ModelNet40 Data ({DATASET_NAME}) ==")
        print(df.transpose())


if __name__ == '__main__':
    main()
