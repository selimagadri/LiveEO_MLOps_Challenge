FROM jupyter/scipy-notebook

COPY scripts ./scripts
COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python ./scripts/main.py --ckpt_path './trained_models/best_model.ckpt' --num_workers 0