FROM tensorflow/tensorflow:1.4.0-gpu-py3
RUN python -m pip install --upgrade pip \
&& pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN git clone https://github.com/tensorflow/nmt/

WORKDIR /app
COPY nmt /app/nmt

RUN mkdir /tmp/nmt_data
RUN mkdir /tmp/nmt_model

RUN bash nmt/scripts/download_iwslt15.sh /app/nmt_data

ENTRYPOINT ["python", "-m", "app/nmt.nmt \
    --src=vi --tgt=en \
    --vocab_prefix=/tmp/nmt_data/vocab  \
    --train_prefix=/tmp/nmt_data/train \
    --dev_prefix=/tmp/nmt_data/tst2012  \
    --test_prefix=/tmp/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu"]
