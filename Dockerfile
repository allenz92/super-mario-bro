FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    SDL_AUDIODRIVER=dummy \
    SDL_VIDEODRIVER=dummy \
    DISPLAY=:0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    build-essential git ca-certificates \
    ffmpeg xvfb \
    libx264-dev libx265-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    libsdl2-2.0-0 libsdl2-dev libsdl2-image-2.0-0 libsdl2-mixer-2.0-0 libsdl2-ttf-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

RUN pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch torchvision torchaudio

COPY . /workspace

CMD ["bash"]
