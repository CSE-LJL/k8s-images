FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN python -m pip install --upgrade pip \
&& pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app
COPY transformer-xl-master.zip /app

RUN unzip /app/transformer-xl-master.zip

CMD ["bash", "/app/transformer-xl-master/pytorch/getdata.sh"]

ENTRYPOINT ["bash", "run_wt103_base.sh", "train", "--work_dir", "/app/transformer-xl-master/pytorch"]
