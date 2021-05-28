#!/usr/bin/env python3
import argparse

from deepclr.config import load_config, Mode
from deepclr.engine import train


def main():
    # parse inputs
    parser = argparse.ArgumentParser(description="Model training.")
    parser.add_argument('config', type=str, help="training configuration (*.yaml)")
    parser.add_argument('--ckpt', default=None, type=str, help="checkpoint for warm restart (*.tar)")
    args = parser.parse_args()

    # print input
    print(f"Configuration: {args.config}")
    if args.ckpt is None:
        mode = Mode.NEW
        print("No checkpoint given")
    else:
        mode = Mode.CONTINUE
        print(f"Checkpoint: {args.ckpt}")

    # load cfg
    cfg = load_config(args.config, mode, args.ckpt)

    train(cfg)


if __name__ == '__main__':
    main()
