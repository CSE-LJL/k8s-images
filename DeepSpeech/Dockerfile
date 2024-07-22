FROM seannaren/deepspeech.pytorch:latest

RUN cd data/ && python common_voice.py && cd ..

ENTRYPOINT ["python", "train.py", "+configs=commonvoice"]
