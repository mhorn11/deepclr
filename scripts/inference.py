#!/usr/bin/env python3
import argparse
from datetime import datetime
import os
import os.path as osp

import torch

from deepclr.config import load_model_config
from deepclr.data import create_input_dataflow
from deepclr.evaluation import load_scenario, Evaluator
from deepclr.models import load_trained_model, ModelInferenceHelper
from deepclr.utils.logging import create_logger


def main():
    # parse inputs
    parser = argparse.ArgumentParser(description="Model inference for evaluation scenario.")
    parser.add_argument('scenario', type=str, help="scenario configuration (*.yaml)")
    parser.add_argument('model_name', type=str, help="model name (subdirectory of MODEL_PATH)")
    parser.add_argument('output_base', type=str, help="base directory for inference output")
    parser.add_argument('--model_path', type=str, default=None,
                        help="alternative model path instead of MODEL_PATH")
    parser.add_argument('--weights', type=str, default='weights.tar', help="model weights (default: weights.tar)")
    args = parser.parse_args()

    # logging
    logger = create_logger('evaluation')

    # load scenario
    logger.info("Loading scenario")
    scene_cfg = load_scenario(args.scenario, with_method=False)

    # filenames and directories
    model_base_path = args.model_path
    if model_base_path is None:
        model_base_path = os.getenv('MODEL_PATH')
        if model_base_path is None:
            raise RuntimeError("Could not get model path from environment variable MODEL_PATH or argument.")

    model_path = osp.join(model_base_path, args.model_name)
    model_file = osp.join(model_path, 'model_config.yaml')
    weights_file = osp.join(model_path, args.weights)

    # read model config
    logger.info("Read model configuration")
    model_cfg = load_model_config(model_file, weights_file)

    # load model
    logger.info("Load model")
    model = load_trained_model(model_cfg)
    model = model.cuda()

    # initialize model inference helper and evaluator
    helper = ModelInferenceHelper(model, is_sequential=scene_cfg.sequential)
    evaluator = Evaluator()

    # create output directory
    output_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = osp.join(args.output_base, f'{output_stamp}_{scene_cfg.name}_{model_cfg.model_type.name}')

    logger.info("Create output directory")
    os.makedirs(output_dir, exist_ok=True)

    # create and store evaluation config
    eval_cfg = scene_cfg.copy()
    eval_cfg.method.name = model_cfg.model_type.name
    eval_cfg.method.params.model_name = args.model_name
    eval_cfg.method.params.model_file = model_file
    eval_cfg.method.params.weights_file = weights_file
    eval_cfg.write_file(osp.join(output_dir, 'scenario.yaml'), invalid=True, internal=True)

    # iterate files
    for data_name, data_file in scene_cfg.data.items():
        logger.info(f"Evaluate '{data_file}'")

        # load data
        df = create_input_dataflow(scene_cfg.dataset_type, data_file, shuffle=False)

        # iterate
        df.reset_state()
        helper.reset_state()
        for i, ds in enumerate(df):
            # print status
            if (i + 1) % 10 == 0:
                logger.info(f"Data point {i + 1}/{len(df)}")

            # prepare data
            template = torch.from_numpy(ds['clouds'][0]).cuda()
            source = torch.from_numpy(ds['clouds'][1]).cuda()
            stamp = ds['timestamps'][0]
            transform_gt = ds['transform']

            # predict with timing
            t_start = torch.cuda.Event(enable_timing=True)
            t_end = torch.cuda.Event(enable_timing=True)
            t_start.record()

            if scene_cfg.sequential:
                if not helper.has_state():
                    helper.predict(template)
                y_pred = helper.predict(source)
            else:
                y_pred = helper.predict(source, template)

            t_end.record()
            torch.cuda.synchronize()

            # get results
            t_pred = t_start.elapsed_time(t_end)

            if y_pred is not None:
                y_pred = y_pred.detach().cpu().numpy()
                transform_pred = model_cfg.label_type.to_matrix(y_pred)
            else:
                transform_pred = None

            # add data to evaluator
            evaluator.add_transforms(data_name, stamp, transform_pred, transform_gt, t_pred)

        del df

    # save results
    logger.info("Store results")
    evaluator.write(output_dir)


if __name__ == '__main__':
    main()
