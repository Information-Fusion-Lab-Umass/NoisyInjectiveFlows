FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

USER root

# Install base stuff
RUN apt-get update && \
    apt-get install -y \
    wget \
    ca-certificates \
    git-core \
    pkg-config \
    tree \
    freetds-dev && \
    # clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Anaconda
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/anaconda.sh
RUN /bin/bash ~/anaconda.sh -b -p /opt/conda
RUN rm ~/anaconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda update conda && conda update anaconda && conda update --all

# Install the latest versions of these python packages
RUN python -m pip install --upgrade pip && \
    pip uninstall numpy -y && \
    pip install --user numpy scipy cython \
    PyQt5 matplotlib tqdm \
    tensorflow-gpu==1.15.2

# Set the base directory
WORKDIR /app
