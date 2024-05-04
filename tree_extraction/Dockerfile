FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
RUN apt update

RUN apt install -y g++ \
         make \
         wget \
         unzip \
         vim \
         git \
         dstat \
         python3-pip

RUN wget \
   https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
   && bash Miniconda3-latest-Linux-x86_64.sh -b -p /conda \
   && rm -f Miniconda3-latest-Linux-x86_64.sh 

ENV PATH=${PATH}:/conda/bin
# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

RUN mkdir /working_dir
WORKDIR /working_dir