FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app
COPY Mobilenetv2.py /app/Mobilenetv2.py

RUN pip install torchvision \
    && pip install matplotlib

ENTRYPOINT [ "python", "-u","/app/Mobilenetv2.py" ]
