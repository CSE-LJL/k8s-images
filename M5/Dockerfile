FROM tensorflow/tensorflow:1.0.1-gpu-py3

RUN git clone https://github.com/philipperemy/very-deep-convnets-raw-waveforms/tree/master

WORKDIR /app
COPY --r very-deep-convnets-raw-waveforms /app/very-deep-convnets-raw-waveforms

RUN pip3 install -r requirements.txt

RUN export CUDA_VISIBLE_DEVICES=1; python3 -u model_run.py m5 > m5.out
