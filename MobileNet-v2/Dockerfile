FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app
COPY latency.py /app/latency.py
COPY models.py /app/models.py
COPY test.py /app/test.py
COPY train.py /app/train.py

RUN pip install torchvision \
    && pip install future==0.18.2

ENTRYPOINT [ "python", "-u","/app/train.py" ]
