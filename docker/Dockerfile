FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
RUN apt update

RUN apt install -y g++ \
         make \
         wget \
         unzip \
         vim \
         git \
         sudo \
         curl \
         htop \
         dstat \
         python3-pip

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs

RUN wget \
   https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
   && bash Miniconda3-latest-Linux-x86_64.sh -b -p /conda \
   && rm -f Miniconda3-latest-Linux-x86_64.sh 

ENV PATH=${PATH}:/conda/bin
# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]
RUN conda init
RUN source ~/.bashrc

RUN mkdir /working_dir
WORKDIR /working_dir
COPY environment.yml environment.yml

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda env create -f environment.yml

ENV TOKENIZERS_PARALLELISM=true