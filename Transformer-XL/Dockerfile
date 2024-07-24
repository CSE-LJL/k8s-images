FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    bash-completion \
    build-essential \
    cmake \
    curl \
    git \
    libboost-all-dev \
    libbz2-dev \
    liblzma-dev \
    locales \
    python3-venv \
    unzip \
    xz-utils \
    wget && \
    # We need to remove it because it's breaking deepspeech install later with \
    # weird errors about setuptools \
    apt-get purge -y python3-xdg && \
    # Install dependencies for audio augmentation \
    apt-get install -y --no-install-recommends libopus0 libsndfile1 && \
    # Try and free some space \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install pytorch-gpu==2.1.0

WORKDIR /app

COPY getdata.sh /app
COPY prep_text8.py /app
COPY data_utils.py /app
COPY eval.py /app
COPY mem_transformer.py /app

COPY run_wt103_base.sh /app
COPY train.py /app

RUN mkdir /app/pytorch
RUN mkdir /app/utils

COPY adaptive_softmax.py /app/utils
COPY data_parallel.py /app/utils
COPY exp_utils.py /app/utils
COPY log_uniform_sampler.py /app/utils
COPY proj_adaptive_softmax.py /app/utils
COPY vocabulary.py /app/utils

RUN bash /app/getdata.sh wikitext-103

RUN bash /app/run_wt103_base.sh train --work_dir /app/pytorch
