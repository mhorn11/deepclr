ARG TAG

FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

# System dependencies
RUN apt update && apt install -q -y --no-install-recommends \
    # personal preference
    zsh \
    # needed for installing python dependencies
    git \
    # opencv
    libglib2.0-0 \
    libgl1-mesa-glx \
    # open3d
    libusb-1.0-0 \
    # python-prctl
    build-essential \
    libcap-dev \
    # GICP
    cmake \
    libgsl-dev \
    # KITTI Devkit
    ghostscript \
    gnuplot \
    texlive-extra-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
RUN python -m pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# CUDA settings
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
ENV TORCH_CUDA_ARCH_LIST="6.1 7.5+PTX"

# External dependencies
COPY extern /tmp/extern
RUN /tmp/extern/install.sh && rm -rf /tmp/extern

# Enable installing into conda for all users
RUN chmod go+w /opt/conda/lib/python3.8/site-packages
