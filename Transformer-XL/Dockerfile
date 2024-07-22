FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN python -m pip install --upgrade pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

COPY getdata.sh /app
COPY prep_text8.py /app
COPY data_utils.py /app
COPY eval.py /app
COPY mem_transformer.py /app

COPY run_wt103_base.sh /app
COPY train.py /app

RUN mkdir /app/pytorch
RUN mkdir /app/utils

COPY adaptive_softmax.py /app/utils
COPY data_parallel.py /app/utils
COPY exp_utils.py /app/utils
COPY log_uniform_sampler.py /app/utils
COPY proj_adaptive_softmax.py /app/utils
COPY vocabulary.py /app/utils

CMD ["bash", "/app/getdata.sh"]

ENTRYPOINT ["bash", "run_wt103_base.sh", "train", "--work_dir", "/app/pytorch"]
