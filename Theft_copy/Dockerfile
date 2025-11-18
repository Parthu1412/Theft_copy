FROM tensorflow/tensorflow:2.15.0-gpu
ENV DEBIAN_FRONTEND=noninteractive
# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    wkhtmltopdf \
    git \
    gcc \
    g++ \
    psmisc \
    redis-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get install -y ffmpeg \
    gcc \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    gir1.2-gtk-3.0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
 
# Set working directory
WORKDIR /home/test
 
# Copy project files
COPY . .
 
 
 
# Upgrade pip and install PyTorch
RUN pip install --upgrade pip
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
 
# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir websocket-client
RUN pip install PyGObject pycairo          
RUN pip install torch==2.9.0
RUN pip install protobuf==3.19.6
RUN pip install tensorflow==2.15.0
 
# Create the startup script directly as part of the CMD
CMD ["bash", "./start_all.sh"]