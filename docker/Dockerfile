ARG BASE=22.05
ARG BASE_IMG=nvcr.io/nvidia/tensorrt:${BASE}-py3

RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install tqdm