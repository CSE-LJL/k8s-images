FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
RUN python -m pip install --upgrade pip \
&& pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \

RUN !gdown --id '1lz0Wtwxsh5YCPdqQ3E3l_nbfJT1N13V8' --output data.zip && !unzip data.zip

WORKDIR /app
COPY Lstm.py /app/Lstm.py

ENTRYPOINT [ "python", "-u","/app/Lstm.py" ]
