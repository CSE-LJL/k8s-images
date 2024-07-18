FROM tensorflow/tensorflow:1.0.0-gpu-py3

WORKDIR /app
COPY very-deep-convnets-raw-waveforms-master.zip /app

RUN unzip /app/very-deep-convnets-raw-waveforms-master.zip

RUN export CUDA_VISIBLE_DEVICES=1

CMD ["pip3", "install", "-r", "/app/very-deep-convnets-raw-waveforms-master/requirements.txt"]

ENTRYPOINT ["python3", "-u", "/app/very-deep-convnets-raw-waveforms-master/model_run.py", "m5", ">", "m5.out"]
