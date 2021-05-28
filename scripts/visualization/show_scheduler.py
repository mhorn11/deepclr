#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt
import torch

from deepclr.config import load_config, Mode
from deepclr.models import build_model
from deepclr.solver import make_optimizer, make_scheduler


def main():
    parser = argparse.ArgumentParser(description="Test scheduler.")
    parser.add_argument('config', type=str)
    parser.add_argument('iterations', type=int)
    args = parser.parse_args()

    # cfg
    cfg = load_config(args.config, mode=Mode.TEST)

    # model, optimizer and scheduler
    model = build_model(cfg.model)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)

    # run
    epoch_list = []
    lr_list = []

    for i in range(args.iterations):
        with torch.no_grad():
            optimizer.step()
            scheduler.step()
            lr = min([grp['lr'] for grp in optimizer.state_dict()['param_groups']])
            epoch_list.append(i)
            lr_list.append(lr)

    # plot
    plt.figure()
    plt.plot(epoch_list, lr_list, '-')
    plt.title('Learning Rate Scheduling')
    plt.xlabel('Iteration')
    plt.ylabel('LR')
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
