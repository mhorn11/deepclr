#!/usr/bin/env python3
import argparse
import os
import os.path as osp

import numpy as np
import torch
import transforms3d as t3d

from deepclr.config import load_model_config
from deepclr.models import load_trained_model, ModelInferenceHelper


def main():
    # parse inputs
    parser = argparse.ArgumentParser(description="Model inference for ModelNet40 data.")
    parser.add_argument('model_name', type=str, help="Model name (directory in MODEL_PATH)")
    args = parser.parse_args()

    # environment variables
    modelnet40_path = os.getenv('MODELNET40_PATH')
    if modelnet40_path is not None:
        modelnet40_directory = osp.join(modelnet40_path, 'original')
        modelnet40_filename = osp.join(modelnet40_directory, 'modelnet40_test.txt')
    else:
        raise RuntimeError("Could not get ModelNet40 path from environment variable MODELNET40_PATH.")

    model_path = os.getenv('MODEL_PATH')
    if model_path is not None:
        model_cfg_filename = osp.join(model_path, args.model_name, 'model_config.yaml')
        model_weights_filename = osp.join(model_path, args.model_name, 'weights.tar')
    else:
        raise RuntimeError("Could not get model path from environment variable MODEL_PATH.")

    # model
    model_cfg = load_model_config(model_cfg_filename, model_weights_filename)
    model = load_trained_model(model_cfg)
    model = model.cuda()

    # inference helper
    helper = ModelInferenceHelper(model, is_sequential=False)

    # modelnet40
    modelnet40_names = [line.rstrip('\n') for line in open(modelnet40_filename)]
    modelnet40_files = [osp.join(modelnet40_directory, name.rpartition('_')[0], f'{name}.txt')
                        for name in modelnet40_names]

    # iterate
    for model_file in modelnet40_files:
        # load source
        source = np.loadtxt(model_file, delimiter=',')[:, :3]

        # random transform
        euler = np.random.rand(3) * 0.04 - 0.02
        t = np.random.rand(3) * 0.2 - 0.1
        r = t3d.euler.euler2mat(*euler)

        # create template
        template = source.dot(r.T) + t

        # prediction
        template_tensor = torch.from_numpy(template.astype(np.float32)).cuda()
        source_tensor = torch.from_numpy(source.astype(np.float32)).cuda()
        y_pred = helper.predict(source_tensor, template_tensor)

        # get matrix
        m_pred = model_cfg.label_type.to_matrix(y_pred.detach().cpu().numpy()) if y_pred is not None else None
        print('Prediction:\n', m_pred)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
