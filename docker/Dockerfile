## Pytracking docker container
#
#
# 

FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ARG username
ARG password
ARG user_id

ENV TZ "Europe/Berlin"
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt upgrade -y && apt install -y apt-utils build-essential software-properties-common wget unzip curl git cmake ca-certificates vim autoconf rsync automake mesa-utils sudo parallel
RUN apt install -y libopencv-dev libboost-all-dev libusb-1.0-0-dev
RUN apt install -y libhdf5-dev hdf5-tools libglew-dev libglfw3-dev libcanberra-gtk-module libboost-program-options-dev libeigen3-dev ffmpeg
RUN apt install -y python3-pip python3-tk python3-dev
RUN apt install ninja-build

RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install torch torchvision
RUN python3 -m pip install pandas pandas pandas opencv-python numpy scipy h5py
RUN python3 -m pip install jupyter jupyterlab matplotlib
RUN python3 -m pip install tb-nightly
RUN python3 -m pip install visdom scikit-image tikzplotlib gdown cython 
RUN python3 -m pip install pycocotools lvis jpeg4py pyqt5-tools

RUN apt install -y ninja-build
RUN apt install -y libturbojpeg
RUN apt install -y x11-apps

RUN useradd -u $user_id -ms /bin/bash $username && usermod -aG sudo $username && echo "$username:$password" | chpasswd

USER $username
WORKDIR /home/$username
