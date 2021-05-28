#!/usr/bin/env python3
import argparse
import os
import os.path as osp
from typing import Optional
import warnings

import numpy as np
import pandas as pd

from deepclr.config import Config
from deepclr.evaluation import Evaluator, load_scenario


SCENARIO_NAME = 'kitti_04_10'
METHOD_NAME = 'DEEPCLR'


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


def evaluate(path: str, scenario: Config) -> None:
    # load
    filenames = [f'{k}.txt' for k in scenario.data.keys()]
    evaluator = Evaluator.read(path, filenames)

    # process
    step_errors = evaluator.get_step_errors()
    total_step_errors = evaluator.get_total_step_errors()

    # convert
    data = {
        seq: {'t_rmse [m]': metrics.mean.translation.rmse,
              'r_rmse [deg]': np.rad2deg(metrics.mean.rotation.rmse)}
        for seq, metrics in step_errors.items()
    }
    df = pd.DataFrame(data)

    # print
    print("== DeepCLR Results on KITTI Odometry ==")
    print(df.transpose())
    print()
    print(f"Average Inference Time: {total_step_errors.mean.time:.2f} ms")


def main():
    # parse inputs
    parser = argparse.ArgumentParser(description="Print table for KITTI odometry evaluation.")
    parser.add_argument('path', type=str, help="base directory of inference output")
    args = parser.parse_args()

    base_path = args.path

    # search scenario
    found = False
    for dirname in sorted(os.listdir(base_path)):
        # check directory
        directory = osp.join(base_path, dirname)
        if not osp.isdir(directory):
            continue

        # load and check scenario
        scenario = load_scenario_from_dir(directory)
        if scenario is not None and scenario.name == SCENARIO_NAME and scenario.method.name == METHOD_NAME:
            evaluate(directory, scenario)
            found = True
            break

    # warning if no scenario was found
    if not found:
        warnings.warn("Could not find scenario.")


if __name__ == '__main__':
    main()
