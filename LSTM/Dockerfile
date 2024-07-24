FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
RUN python -m pip install --upgrade pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install jieba

WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN mkdir /app/data

COPY dev.csv /app/data
COPY train.csv /app/data
COPY test.csv /app/data

COPY config.py /app
COPY entity.py /app
COPY main.py /app

ENTRYPOINT [ "python", "-u","/app/main.py" ]
