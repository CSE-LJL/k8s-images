FROM seannaren/deepspeech.pytorch:latest

RUN cd data/ && python librispeech.py && cd ..

ENTRYPOINT ["python", "train.py", "+configs=librispeech"]
