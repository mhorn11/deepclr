#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

KITTI_DEVKIT_DOWNLOAD_PATH="https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_odometry.zip"

# Pointnet2.PyTorch
echo "Preparing Pointnet2.PyTorch"
cd "${SCRIPT_DIR}" || exit
git submodule update --init Pointnet2.PyTorch || exit
cd "${SCRIPT_DIR}/Pointnet2.PyTorch" || exit
git reset HEAD --hard || exit
git clean -fdx || exit
git apply "${SCRIPT_DIR}/pointnet2.patch" || exit

# GICP
echo "Preparing GICP"
cd "${SCRIPT_DIR}/gicp" || exit
git submodule update --init gicp || exit

# KITTI Devkit
echo "Preparing KITTI Devkit"
wget ${KITTI_DEVKIT_DOWNLOAD_PATH} -O /tmp/devkit_odometry.zip || exit
unzip -o /tmp/devkit_odometry.zip -d /tmp/devkit_odometry || exit
rm -r "${SCRIPT_DIR}/kitti_devkit/cpp"
mv -f /tmp/devkit_odometry/devkit/* "${SCRIPT_DIR}/kitti_devkit" || exit
cd "${SCRIPT_DIR}/kitti_devkit/cpp" || exit
patch -p1 < ../../kitti_devkit.patch || exit