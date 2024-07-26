FROM yangho/tensorflow1.8-gpu:cuda9.0-cudnn7.0.5.15
#RUN python3 -m pip install --upgrade pip
#RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install numpy
RUN pip install pandas


RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    bash-completion \
    build-essential \
    cmake \
    curl \
    git \
   # libboost-all-dev \
   # libbz2-dev \
   # liblzma-dev \
   # locales \
   # python3-venv \
    unzip \
    xz-utils \
    wget && \
    # We need to remove it because it's breaking deepspeech install later with \
    # weird errors about setuptools \
   # apt-get purge -y python3-xdg && \
    # Install dependencies for audio augmentation \
   # apt-get install -y --no-install-recommends libopus0 libsndfile1 && \
   # apt install -y  net-tools python3-pip \
    # Try and free some space \
    rm -rf /var/lib/apt/lists/*

#RUN python -m pip install numpy
#RUN pip install pandas

WORKDIR /app

#RUN git clone https://github.com/tensorflow/nmt
#COPY libcudnn7_7.6.5.32-1+cuda9.0_amd64.deb /app

#RUN apt-get install -y  /app/libcudnn7_7.6.5.32-1+cuda9.0_amd64.deb
RUN mkdir /app/nmt_data
RUN mkdir /app/nmt_model
COPY nmt.zip /app
RUN unzip /app/nmt.zip
#CMD ["python", "-m", "pip", "install", "numpy"]
RUN bash /app/nmt/scripts/download_iwslt15.sh /app/nmt_data
RUN export PYTHONPATH=$PYTHONPATH:/app/nmt
CMD ["python", "/app/nmt/nmt.py", "--src", "vi", "--tgt", "en", "--vocab_prefix", "/app/nmt_data/vocab", "--train_prefix", "/app/nmt_data/train", "--dev_prefix", "/app/nmt_data/tst2012", "--test_prefix", "/app/nmt_data/tst2013", "--out_dir", "/app/nmt_model", "--num_train_steps", "12000", "--steps_per_stats", "100", "--num_layers", "2", "--num_units", "128", "--dropout", "0.2", "--metrics", "bleu"]
