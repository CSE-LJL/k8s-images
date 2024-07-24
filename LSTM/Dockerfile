FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
RUN python -m pip install --upgrade pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install jieba
RUN python -m pip install torchtext==0.10.1

WORKDIR /app
#ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN mkdir /app/data
RUN mkdir /app/save

COPY dev.csv /app/data
COPY train.csv /app/data
COPY test.csv /app/data

COPY config.py /app
COPY entity.py /app
COPY main.py /app

ENTRYPOINT [ "python", "-u","/app/main.py" ]
