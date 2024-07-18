FROM tensorflow/tensorflow:1.0.0-gpu-py3

WORKDIR /app

RUN git clone https://github.com/vivek081166/raw-audio-deep-learning/tree/master

RUN export CUDA_VISIBLE_DEVICES=1

CMD ["pip3", "install", "-r", "/app/raw-audio-deep-learning/requirements.txt"]

CMD ["python", "-u", "/app/raw-audio-deep-learning/process_data.py"]

ENTRYPOINT ["python3", "-u", "/app/raw-audio-deep-learning/model.py"]
