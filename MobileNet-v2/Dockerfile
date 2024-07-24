FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app
COPY latency.py /app/latency.py
COPY models.py /app/models.py
COPY test.py /app/test.py
COPY train.py /app/train.py
RUN mkdir /app/trained_models
COPY Mobilenet_v3_small_1.0_best.pth /app/trained_models

RUN pip install torchvision \
    && pip install future==0.18.2

ENTRYPOINT [ "python", "-u","/app/train.py" ]
