FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu16.04

#- Upgrade system and install dependencies -------------------------------------
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
                    build-essential \
                    cmake \
                    git \
                    unzip \
                    libboost-program-options-dev \
                    libboost-system-dev \
                    libboost-test-dev \
                    libboost-thread-dev \
                    libbz2-dev \
                    libeigen3-dev \
                    liblzma-dev \
                    libsndfile1 \
                    python3 \
                    python3-dev \
                    python3-pip \
                    python3-setuptools \
                    python3-wheel \
                    sudo \
                    vim && \
    rm -rf /var/lib/apt/lists

RUN pip install -r requirements.txt

WORKDIR /app
RUN git clone --recursive https://github.com/parlance/ctcdecode.git
RUN cd ctcdecode && pip install .
RUN COPY *.py /app
ENTRYPOINT ["python", "/app/main.py", "-cp", "conf", "-cn", "configs"]
