FROM tensorflow/tensorflow:1.0.1-gpu-py3
RUN python -m pip install --upgrade pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

RUN export CUDA_VISIBLE_DEVICES=1

CMD ["pip3", "install", "-r", "/app/requirements.txt"]

CMD ["python", "-u", "/app/process_data.py"]

ENTRYPOINT ["python3", "-u", "/app/model.py"]
