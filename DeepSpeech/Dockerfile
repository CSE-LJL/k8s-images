FROM seannaren/deepspeech.pytorch:latest

RUN cd data/ && python an4.py && cd ..

RUN python train.py +configs=an4
