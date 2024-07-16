FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN python -m pip install --upgrade pip \
&& pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN git clone https://github.com/kimiyoung/transformer-xl/tree/master

WORKDIR /app
COPY --r transformer-xl/pytorch /app/transformer-xl
COPY transformer-xl/getdata.sh /app/transformer-xl

RUN bash getdata.sh

RUN bash run_wt103_base.sh train --work_dir /app
