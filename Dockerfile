FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

WORKDIR /workspace

ENV PYTHON_VERSION=3.12.9
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl \
    git \
    g++ \
    gnupg \
    libatomic1 \
    libegl1 \  
    libbz2-dev \
    libffi-dev \
    libgl1 \
    libgomp1 \
    lsb-release \
    liblzma-dev \
    libssl-dev \
    make \
    software-properties-common \
    tzdata \
    unzip \
    wget \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

######################## Python installation #########################

# Download Python source code from official site and build it
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar -zxvf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure --enable-optimizations && make && make install && \
    cd .. && \
    rm Python-$PYTHON_VERSION.tgz && \
    rm -r Python-$PYTHON_VERSION

#################### Create Virtual Environment ######################

ENV VIRTUAL_ENV=/workspace/venv
RUN python3.12 -m venv $VIRTUAL_ENV

# by adding the venv to the search path, we avoid activating it in each command
# see https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


######## Install PyTorch and packages that depend on PyTorch #########

RUN python -m pip install --no-cache torch torchvision --index-url https://download.pytorch.org/whl/cu128

RUN git clone https://github.com/ai4trees/tree_species_seg.git
RUN python -m pip install --no-cache ./tree_species_seg[dev]
