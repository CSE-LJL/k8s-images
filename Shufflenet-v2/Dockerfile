FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app
RUN mkdir /app/data
COPY Shufflenetv2.py /app/Shufflenetv2.py

RUN pip install torchvision \
    && pip install torchstat

ENTRYPOINT [ "python", "-u","/app/Shufflenetv2.py" ]
