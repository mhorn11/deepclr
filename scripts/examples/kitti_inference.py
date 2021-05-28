#!/usr/bin/env python3
import argparse
import os
import os.path as osp

import numpy as np
import pykitti
import torch

from deepclr.config import load_model_config
from deepclr.models import load_trained_model, ModelInferenceHelper


def main():
    # parse inputs
    parser = argparse.ArgumentParser(description="Model inference for KITTI sequence.")
    parser.add_argument('model_name', type=str, help="Model name (directory in MODEL_PATH)")
    parser.add_argument('sequence', type=str, help="KITTI sequence")
    args = parser.parse_args()

    # environment variables
    kitti_path = os.getenv('KITTI_PATH')
    if kitti_path is not None:
        kitti_base_path = osp.join(kitti_path, 'original')
    else:
        raise RuntimeError("Could not get KITTI path from environment variable KITTI_PATH.")

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
    helper = ModelInferenceHelper(model, is_sequential=True)

    # kitti
    kitti = pykitti.odometry(kitti_base_path, args.sequence)

    # iterate
    for cloud in kitti.velo:
        # prediction
        cloud_tensor = torch.from_numpy(cloud.astype(np.float32)).cuda()
        y_pred = helper.predict(cloud_tensor)

        # get matrix
        m_pred = model_cfg.label_type.to_matrix(y_pred.detach().cpu().numpy()) if y_pred is not None else None
        print('Prediction:\n', m_pred)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
