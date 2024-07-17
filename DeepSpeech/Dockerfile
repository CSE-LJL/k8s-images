FROM seannaren/deepspeech.pytorch:latest

RUN cd data/ && python an4.py && cd ..

ENTRYPOINT ["python", "train.py", "+configs=an4"]
