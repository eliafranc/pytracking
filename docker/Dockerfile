## Pytracking docker container
#
#
# 

FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ARG username
ARG password
ARG user_id
ARG DEBIAN_FRONTEND=noninteractive

ENV TZ "Europe/Berlin"
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt update -y && apt upgrade -y && apt install -y \
    apt-utils \
    build-essential \
    software-properties-common \
    wget \
    unzip \
    curl \
    git \
    cmake \
    ca-certificates\
    vim \
    autoconf \
    rsync \
    automake \
    mesa-utils \
    sudo \
    parallel \
    libopencv-dev \
    libboost-all-dev \
    libusb-1.0-0-dev \
    libhdf5-dev \
    hdf5-tools \
    libglew-dev \
    libglfw3-dev \
    libcanberra-gtk-module \
    libboost-program-options-dev \
    libeigen3-dev \
    ffmpeg \
    python3-pip \
    python3-tk \
    python3-dev \
    ninja-build \
    libturbojpeg \
    x11-apps

RUN python3 -m pip install --upgrade pip && python3 -m pip install \
    torch \
    torchvision \
    pandas \
    opencv-python \
    numpy \
    scipy \
    h5py \
    jupyter \
    jupyterlab \
    matplotlib \
    tb-nightly \
    visdom \
    scikit-image \
    tikzplotlib \
    gdown \
    cython \ 
    pycocotools \
    lvis \
    jpeg4py \
    pyqt5-tools    

RUN useradd -u $user_id -ms /bin/bash $username && usermod -aG sudo $username \
    && echo "$username:$password" | chpasswd

USER $username
WORKDIR /home/$username
