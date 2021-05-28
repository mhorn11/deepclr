#!/usr/bin/env python3
import argparse
import os
import os.path as osp
from typing import Dict, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np

from deepclr.config import Config
from deepclr.evaluation import Evaluator, load_scenario


SCENARIO_NAME = 'modelnet40_unseen'
DATASET_NAME_TEMPLATE = 'test_unseen_{noise:0.2f}'
NOISE_LEVELS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]


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
    step_errors = evaluator.get_step_errors()

    # convert
    rotation_errors = []
    translation_errors = []
    for noise in NOISE_LEVELS:
        dataset_name = DATASET_NAME_TEMPLATE.format(noise=noise)
        if dataset_name not in step_errors:
            raise RuntimeError(f"Dataset '{dataset_name}' not found for method '{scenario.method.name}'")

        rotation_errors.append(np.rad2deg(step_errors[dataset_name].mean.rotation.chordal))
        translation_errors.append(step_errors[dataset_name].mean.translation.kitti)

    return {'r': rotation_errors, 't': translation_errors}


def plot_errors(data: Dict) -> None:
    # figures
    fig_r = plt.figure()
    fig_t = plt.figure()

    ax_r = fig_r.gca()
    ax_t = fig_t.gca()

    # data
    methods = []
    for method, errors in data.items():
        methods.append(method)
        ax_r.plot(NOISE_LEVELS, errors['r'], '-')
        ax_t.plot(NOISE_LEVELS, errors['t'], '-')

    # legend
    ax_r.legend(methods)
    ax_t.legend(methods)

    # axes
    ax_r.set_xlabel('Standard Deviation of Gaussian Noise')
    ax_t.set_xlabel('Standard Deviation of Gaussian Noise')
    ax_r.set_ylabel('Rotation Error [deg]')
    ax_t.set_ylabel('Translation Error [m]')

    # grid
    ax_r.grid()
    ax_t.grid()

    plt.show()


def main():
    # parse inputs
    parser = argparse.ArgumentParser(description="Generate plots for ModelNet40 evaluation.")
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
        plot_errors(data)


if __name__ == '__main__':
    main()
