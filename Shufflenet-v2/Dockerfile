FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app
COPY Shufflenetv2.py /app/Shufflenetv2.py

RUN pip install torchvision \
    && pip install matplotlib

ENTRYPOINT [ "python", "-u","/app/Shufflenetv2.py" ]
