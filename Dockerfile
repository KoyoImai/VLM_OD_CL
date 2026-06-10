FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# --- System packages ---
RUN apt-get update && apt-get install -y \
    wget curl git git-lfs \
    tmux vim build-essential ninja-build \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Miniconda (Python 3.10) ---
ENV PATH=/opt/conda/bin:${PATH}
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

# --- PyTorch 2.1.2 + CUDA 12.1 wheels (CUDA 12.8 driver は後方互換) ---
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# --- OpenMMLab: mmengine / mmcv / mmdetection ---
RUN pip install --no-cache-dir mmengine==0.10.3

# mmcv prebuilt wheel (cu121 / torch2.1.0)
RUN pip install --no-cache-dir \
    mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html

# # mmdetection をソースからインストール (GLIP・Grounding DINO のコンフィグを含む)
# RUN git clone https://github.com/open-mmlab/mmdetection.git /workspace/kouyou/mmdetection \
#     && cd /workspace/kouyou/mmdetection \
#     && pip install --no-cache-dir -e .

# --- GLIP / MM-Grounding DINO 追加依存パッケージ ---
# BERT等の言語モデル, テキスト処理, Vision Transformer 関連
RUN pip install --no-cache-dir \
    transformers==4.38.2 \
    tokenizers>=0.15.0 \
    spacy \
    nltk \
    scipy \
    timm \
    einops \
    ftfy \
    regex \
    pycocoevalcap \
    pycocotools


RUN python -m spacy download en_core_web_sm || true

# --- Hugging Face キャッシュ先 (ホストの /data1/kouyou/HuggingFace をマウント推奨) ---
ENV HF_HOME=/workspace/kouyou/datasets/HuggingFace
ENV TRANSFORMERS_CACHE=/workspace/kouyou/datasets/HuggingFace/hub

WORKDIR /workspace
RUN chmod -R 777 /workspace

EXPOSE 7979


# Dockerコンテナ作成後，以下を実行
# cd /workspace/kouyou/mmdetection
# pip install --no-cache-dir -e .
# pip install --no-cache-dir "numpy<2"
# pip install --no-cache-dir "opencv-python==4.9.0.80"
# pip install --no-cache-dir fairscale jsonlines