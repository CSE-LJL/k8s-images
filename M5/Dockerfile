FROM tensorflow/tensorflow:1.0.1-gpu-py3
RUN python -m pip install --upgrade pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get install unzip

RUN apt-get install git

WORKDIR /app

RUN git clone https://github.com/vivek081166/raw-audio-deep-learning/tree/master

RUN export CUDA_VISIBLE_DEVICES=1

CMD ["pip3", "install", "-r", "/app/raw-audio-deep-learning/requirements.txt"]

CMD ["python", "-u", "/app/raw-audio-deep-learning/process_data.py"]

ENTRYPOINT ["python3", "-u", "/app/raw-audio-deep-learning/model.py"]
