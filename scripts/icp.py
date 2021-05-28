#!/usr/bin/env python3
import argparse
from datetime import datetime
import os
import os.path as osp
import time

from deepclr.data import create_input_dataflow
from deepclr.evaluation import load_scenario, Evaluator
from deepclr.icp import ICPAlgorithm, ICPRegistration
from deepclr.utils.parsing import ParseEnum
from deepclr.utils.logging import create_logger


def main():
    # parse inputs
    parser = argparse.ArgumentParser(description="ICP registration for evaluation scenario.")
    parser.add_argument('scenario', type=str, help="scenario configuration (*.yaml)")
    parser.add_argument('algorithm', action=ParseEnum, enum_type=ICPAlgorithm, help="ICP algorithm type")
    parser.add_argument('output_base', type=str, help="base directory for inference output")
    parser.add_argument('--max-distance', type=float, default=1.0, help="maximal distance for ICP (default: 1.0)")
    parser.add_argument('--neighbor-radius', type=float, default=1.0,
                        help="neighbor radius (e.g. for ICP plane) (default: 1.0)")
    parser.add_argument('--max-nn', type=int, default=30, help="maximal number of neighbors (default: 30)")
    args = parser.parse_args()

    # logging
    logger = create_logger('evaluation')

    # load scenario
    logger.info("Loading scenario")
    scene_cfg = load_scenario(args.scenario, with_method=False)

    # create registration and evaluator
    registration = ICPRegistration(args.algorithm, max_distance=args.max_distance, neighbor_radius=args.neighbor_radius,
                                   max_nn=args.max_nn)
    evaluator = Evaluator()

    # create output directory
    output_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = osp.join(args.output_base, f'{output_stamp}_{scene_cfg.name}_{args.algorithm.name}')

    logger.info("Create output directory")
    os.makedirs(output_dir, exist_ok=True)

    # create and store evaluation config
    eval_cfg = scene_cfg.copy()
    eval_cfg.method.name = args.algorithm.name
    eval_cfg.method.params.max_distance = args.max_distance
    eval_cfg.method.params.neighbor_radius = args.neighbor_radius
    eval_cfg.method.params.max_nn = args.max_nn
    eval_cfg.write_file(osp.join(output_dir, 'scenario.yaml'), invalid=True, internal=True)

    # iterate files
    for data_name, data_file in scene_cfg.data.items():
        logger.info(f"Evaluate '{data_file}'")

        # load data
        df = create_input_dataflow(scene_cfg.dataset_type, data_file, shuffle=False)

        # iterate
        df.reset_state()
        for i, ds in enumerate(df):
            # print status
            if (i + 1) % 10 == 0:
                logger.info(f"Data point {i + 1}/{len(df)}")

            # prepare data
            template = ds['clouds'][0][:, :3]
            source = ds['clouds'][1][:, :3]
            stamp = ds['timestamps'][0]
            transform_gt = ds['transform']

            template = registration.prepare(template)
            source = registration.prepare(source)

            # register with timing
            t_start = time.time()

            transform_pred = registration.register(template, source)

            t_end = time.time()
            t_reg = (t_end - t_start) * 1000

            # add data to evaluator
            evaluator.add_transforms(data_name, stamp, transform_pred, transform_gt, t_reg)

        del df

    # save results
    logger.info("Store results")
    evaluator.write(output_dir)


if __name__ == '__main__':
    main()
