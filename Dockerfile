# Imagem reproduzivel para treino e inferencia.
# Base: PyTorch oficial com CUDA 12.1 + Python 3.10
# Em maquina sem GPU: usar tag :latest do pytorch/pytorch (ou ajustar pra cpu).

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONHASHSEED=42

# deps de sistema (opencv precisa de libgl, ffmpeg ajuda em video)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        wget \
        unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# instala deps em camada separada pra cache
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# copia o codigo (datasets e runs/ ficam fora — montar como volumes)
COPY *.py /workspace/
COPY *.md /workspace/
COPY *.sh /workspace/
COPY .gitignore /workspace/

# diretorios padrao para volumes (montar com -v <host>:<container>)
RUN mkdir -p /workspace/datasets /workspace/runs /workspace/captures

# Como usar (a partir da raiz do projeto):
#   docker build -t money-pnp .
#
#   # Treino com GPU
#   docker run --gpus all --rm -it \
#       -v $PWD/datasets:/workspace/datasets \
#       -v $PWD/runs:/workspace/runs \
#       -v $PWD/captures:/workspace/captures \
#       -e WANDB_API_KEY=$WANDB_API_KEY \
#       money-pnp python train_v2.py
#
#   # Eval
#   docker run --gpus all --rm -v $PWD/datasets:/workspace/datasets -v $PWD/runs:/workspace/runs \
#       money-pnp python evaluate_model.py --model runs/detect/v2/weights/best.pt
#
# Webcam em tempo real NAO roda dentro do container (sem acesso direto a USB no Windows host).
# Para realtime, rode python realtime_test.py direto no host.

CMD ["python", "-c", "import torch; print('cuda?', torch.cuda.is_available()); print('Pronto. Use python <script>.py')"]
