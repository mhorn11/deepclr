#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Pointnet2.PyTorch
echo "Installing Pointnet2.PyTorch"
cd "${SCRIPT_DIR}/Pointnet2.PyTorch" || exit
python -m pip install . || exit

# GICP
echo "Installing GICP"
cd "${SCRIPT_DIR}/gicp" || exit
python -m pip install . || exit

# KITTI Devkit
echo "Building KITTI Devkit"
cd "${SCRIPT_DIR}/kitti_devkit" || exit
python -m pip install . || exit