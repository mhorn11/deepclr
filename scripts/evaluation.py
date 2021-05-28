#!/usr/bin/env python3
import argparse
from collections import OrderedDict
import os
import os.path as osp
from typing import Any, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

from deepclr.config import Config
from deepclr.evaluation import MetricsContainer, Evaluator, load_scenario


SAVEFIG_ARGS = {'bbox_inches': 'tight', 'pad_inches': 0}


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


def create_dir(*args: str) -> str:
    directory = osp.join(*args)
    os.makedirs(directory, exist_ok=True)
    return directory


def get_error_dict(name: str, error: MetricsContainer, with_time: bool, method: Optional[str] = None,
                   params: Optional[str] = None, is_normalized: bool = False) -> OrderedDict:
    data: List[Tuple[str, Any]] = [('name', name)]

    if method is not None:
        data.append(('method', method))
    if params is not None:
        data.append(('params', params))

    if is_normalized:
        t_factor = 100
        t_unit = '%'
        r_unit = 'deg/m'
    else:
        t_factor = 1
        t_unit = 'm'
        r_unit = 'deg'

    data.extend([
        (f't_kitti_mean [{t_unit}]', error.mean.translation.kitti * t_factor),
        (f't_kitti_std [{t_unit}]', error.std.translation.kitti * t_factor),
        (f't_kitti_max [{t_unit}]', error.max.translation.kitti * t_factor),
        (f't_rmse_mean [{t_unit}]', error.mean.translation.rmse * t_factor),
        (f't_rmse_std [{t_unit}]', error.std.translation.rmse * t_factor),
        (f't_rmse_max [{t_unit}]', error.max.translation.rmse * t_factor),
        (f'r_kitti_mean [{r_unit}]', np.rad2deg(error.mean.rotation.kitti)),
        (f'r_kitti_std [{r_unit}]', np.rad2deg(error.std.rotation.kitti)),
        (f'r_kitti_max [{r_unit}]', np.rad2deg(error.max.rotation.kitti)),
        (f'r_rmse_mean [{r_unit}]', np.rad2deg(error.mean.rotation.rmse)),
        (f'r_rmse_std [{r_unit}]', np.rad2deg(error.std.rotation.rmse)),
        (f'r_rmse_max [{r_unit}]', np.rad2deg(error.max.rotation.rmse)),
        (f'r_chordal_mean [{r_unit}]', np.rad2deg(error.mean.rotation.chordal)),
        (f'r_chordal_std [{r_unit}]', np.rad2deg(error.std.rotation.chordal)),
        (f'r_chordal_max [{r_unit}]', np.rad2deg(error.max.rotation.chordal))
    ])

    if with_time:
        data.extend([
            ('time_mean [ms]', error.mean.time),
            ('time_std [ms]', error.std.time),
            ('time_max [ms]', error.max.time)
        ])

    return OrderedDict(data)


def evaluate_single(base_path: str, scenario: Config) -> Evaluator:
    # filenames from scenario
    filenames = [f'{k}.txt' for k in scenario.data.keys()]

    # load data
    evaluator = Evaluator.read(base_path, filenames)

    # create output directories
    output_dir = create_dir(base_path, 'evaluation')

    # step errors
    step_errors = []
    for name, seg_err in evaluator.get_step_errors().items():
        step_errors.append(get_error_dict(name, seg_err, with_time=True, is_normalized=False))
    step_errors.append(get_error_dict('TOTAL', evaluator.get_total_step_errors(), with_time=True, is_normalized=False))

    step_df = pd.DataFrame.from_dict(step_errors)
    step_df.to_csv(osp.join(output_dir, 'step_errors.csv'), index=False)

    # segment errors
    if scenario.sequential:
        # errors
        segment_errors = []
        for name, seg_err in evaluator.get_segment_errors().items():
            segment_errors.append(get_error_dict(name, seg_err, with_time=False, is_normalized=True))
        segment_errors.append(get_error_dict('TOTAL', evaluator.get_total_segment_errors(),
                                             with_time=False, is_normalized=True))

        segment_df = pd.DataFrame.from_dict(segment_errors)
        segment_df.to_csv(osp.join(output_dir, 'segment_errors.csv'), index=False)

        # plots
        fig_bars = evaluator.plot_segment_error_bars()
        fig_bars.savefig(osp.join(output_dir, 'segment_errors.png'), **SAVEFIG_ARGS)
        fig_bars.savefig(osp.join(output_dir, 'segment_errors.pdf'), **SAVEFIG_ARGS)

        eot_dir = create_dir(output_dir, 'plot_eot')
        kitti_dir = create_dir(output_dir, 'plot_error')
        seq_dir = create_dir(output_dir, 'plot_path')
        seq2d_dir = create_dir(output_dir, 'plot_path2d')

        for name, fig_eot in evaluator.plot_error_over_time().items():
            fig_eot.savefig(osp.join(eot_dir, f'{name}.png'), **SAVEFIG_ARGS)
            fig_eot.savefig(osp.join(eot_dir, f'{name}.pdf'), **SAVEFIG_ARGS)

        for name, fig_kitti in evaluator.plot_kitti_errors().items():
            fig_kitti.savefig(osp.join(kitti_dir, f'{name}.png'), **SAVEFIG_ARGS)
            fig_kitti.savefig(osp.join(kitti_dir, f'{name}.pdf'), **SAVEFIG_ARGS)

        for name, fig_seq in evaluator.plot_sequences().items():
            fig_seq.savefig(osp.join(seq_dir, f'{name}.png'), **SAVEFIG_ARGS)
            fig_seq.savefig(osp.join(seq_dir, f'{name}.pdf'), **SAVEFIG_ARGS)

        for name, fig_seq2d in evaluator.plot_sequences_2d().items():
            fig_seq2d.savefig(osp.join(seq2d_dir, f'{name}.png'), **SAVEFIG_ARGS)
            fig_seq2d.savefig(osp.join(seq2d_dir, f'{name}.pdf'), **SAVEFIG_ARGS)

    return evaluator


def evaluate_multi(base_path: str, scenario_name: str) -> None:
    step_errors = []
    segment_errors = []

    found = False
    for dirname in sorted(os.listdir(base_path)):
        # check directory
        directory = osp.join(base_path, dirname)
        if not osp.isdir(directory):
            continue

        # load and check scenario
        scenario = load_scenario_from_dir(directory)
        if scenario is None or scenario.name != scenario_name:
            continue
        found = True

        # single evaluation
        evaluator = evaluate_single(directory, scenario)

        # multi evaluation
        params_str = ', '.join([f'{k}={v}' for k, v in scenario.method.params.items()])

        step_errors.append(get_error_dict(dirname, evaluator.get_total_step_errors(), with_time=True,
                                          method=scenario.method.name, params=params_str, is_normalized=False))
        if scenario.sequential:
            segment_errors.append(get_error_dict(dirname, evaluator.get_total_segment_errors(), with_time=False,
                                                 method=scenario.method.name, params=params_str, is_normalized=True))

    # export evaluation
    if found:
        scenario_output_path = osp.join(base_path, 'evaluation', scenario_name)
        os.makedirs(scenario_output_path, exist_ok=True)

        if len(step_errors) > 0:
            step_df = pd.DataFrame.from_dict(step_errors)
            step_df.to_csv(osp.join(scenario_output_path, f'{scenario_name}_step_errors.csv'), index=False)

        if len(segment_errors) > 0:
            segment_df = pd.DataFrame.from_dict(segment_errors)
            segment_df.to_csv(osp.join(scenario_output_path, f'{scenario_name}_segment_errors.csv'), index=False)

    else:
        warnings.warn(f"No evaluation found for scenario '{scenario_name}'")


def main():
    # parse inputs
    parser = argparse.ArgumentParser(description="Run evaluation on predicted transformations.")
    parser.add_argument('path', type=str, help="direct or base directory of inference or icp output")
    parser.add_argument('--scenario', type=str, default=None, help="evaluation scenario")
    args = parser.parse_args()

    if args.scenario is None:
        scenario = load_scenario_from_dir(args.path)
        if scenario is not None:
            evaluate_single(args.path, scenario)
    else:
        evaluate_multi(args.path, args.scenario)


if __name__ == '__main__':
    main()
