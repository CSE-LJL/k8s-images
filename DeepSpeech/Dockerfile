FROM quay.io/codait/max-speech-to-text-converter

WORKDIR /app

COPY UrbanSound8K.tar.gz /app

RUN tar -zxvf /app/UrbanSound8K.tar.gz

RUN ["curl", "-F", "audio=@/app/UrbanSound8K/audio/fold1/177621-0-0-100.wav", "-X", "POST", "http://localhost:5000/model/predict"]
RUN ["curl", "-F", "audio=@/app/UrbanSound8K/audio/fold1/180937-7-3-2.wav", "-X", "POST", "http://localhost:5000/model/predict"]
RUN ["curl", "-F", "audio=@/app/UrbanSound8K/audio/fold1/180937-7-3-6.wav", "-X", "POST", "http://localhost:5000/model/predict"]
RUN ["curl", "-F", "audio=@/app/UrbanSound8K/audio/fold1/176787-5-0-24.wav", "-X", "POST", "http://localhost:5000/model/predict"]
