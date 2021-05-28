#!/usr/bin/env python3
import argparse

import torch

from deepclr.config import Config, load_config, Mode
from deepclr.data import make_data_loader
from deepclr.models import build_model, ModelInferenceHelper
from deepclr.utils.logging import create_logger
from deepclr.utils.tensor import prepare_tensor


def timing(cfg: Config, sequential: bool) -> None:
    # load model
    model = build_model(cfg.model)
    model.to(cfg.device)
    model.eval()

    # model inference helper
    helper = ModelInferenceHelper(model, is_sequential=sequential)

    # data loader
    data_loader = make_data_loader(cfg, is_train=False, batch_size=1)

    for batch in data_loader:
        # prepare data
        x = prepare_tensor(batch['x'], device=cfg.device)
        template = x[0, ...]
        source = x[1, ...]

        # predict with timing
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        t_start.record()

        if sequential:
            if not helper.has_state():
                helper.predict(template)
            helper.predict(source)
        else:
            helper.predict(source, template)

        t_end.record()
        torch.cuda.synchronize()

        # get results
        print(t_start.elapsed_time(t_end))


def main():
    parser = argparse.ArgumentParser(description="Test inference time with untrained model.")
    parser.add_argument('config', type=str, help="training configuration (*.yaml)")
    parser.add_argument('--sequential', action='store_true', help="activate sequential inference")
    args = parser.parse_args()

    # read cfg
    cfg = load_config(args.config, Mode.TEST)

    # setup logger
    logger = create_logger(name='timing')
    logger.info(cfg.dump())

    # run timing test
    with torch.no_grad():
        timing(cfg, args.sequential)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
