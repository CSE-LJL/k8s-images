FROM tensorflow/tensorflow:1.0.1-gpu-py3
RUN python -m pip install --upgrade pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

COPY constants.py /app
COPY model.py /app
COPY models.py /app
COPY process_data.py /app
COPY requirements.txt /app

RUN mkdir /app/audio/set_a
RUN mkdir /app/audio/set_b

COPY audio.set_a/*.wav /app/audio/set_a
COPY audio.set_b/*.wav /app/audio/set_b

RUN export CUDA_VISIBLE_DEVICES=1

CMD ["pip3", "install", "-r", "/app/requirements.txt"]

CMD ["python", "-u", "/app/process_data.py"]

ENTRYPOINT ["python3", "-u", "/app/model.py"]
