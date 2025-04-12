# For LSV A100s server
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set path to CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

ENV DEBIAN_FRONTEND=noninteractive

# Install additional programs
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    htop \
    gnupg \
    curl \
    ca-certificates \
    vim \
    tmux \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    python3-wheel \
    cuda-command-line-tools-12-2 \
    cuda-cudart-dev-12-2 \
    cuda-cudart-12-2 \
    nvidia-utils-450 \
    && rm -rf /var/lib/apt/lists/*

RUN nvcc --version || echo "nvcc not found"
RUN ldconfig -p | grep cuda || echo "CUDA libraries not found"
RUN nvidia-smi || echo "nvidia-smi not found"

# Install Python dependencies
RUN pip3 install --upgrade pip setuptools wheel

# Explicitly install Python packages and check CUDA
#RUN python -m pip install --upgrade pip "setuptools<71"
#RUN python -m pip install torch

RUN python3 -m pip install --upgrade pip "setuptools==69.5.1"

# Install specific version of PyTorch
RUN pip3 install torch==2.4.0

RUN git clone https://github.com/Dao-AILab/causal-conv1d.git /opt/causal-conv1d
#RUN git clone https://github.com/rozariwang/causal-conv1d.git /opt/causal-conv1d

# Install the missing packaging library
RUN python3 -m pip install packaging

RUN cd /opt/causal-conv1d && pip install -v . --no-cache-dir --no-build-isolation 



# Install Python dependencies
RUN python3 -m pip install \
    accelerate \
    wandb \
    optuna \
    pandas \
    scikit-learn \
    transformers \
    plotly \
    matplotlib \
    rdkit-pypi \
    datasets \
    ninja
    # \
    #causal-conv1d

#RUN python3 -m pip install -v mamba-ssm

# Clone the mamba repository
RUN git clone https://github.com/state-spaces/mamba.git /opt/mamba
#RUN git clone https://github.com/rozariwang/mamba.git /opt/mamba
# Install mamba from the cloned repository
RUN cd /opt/mamba && pip install -v . --no-cache-dir --no-build-isolation

# Uninstall and Reinstall mamba-ssm with no cache
RUN pip3 uninstall mamba-ssm -y
RUN pip3 install mamba-ssm --no-cache-dir

#RUN python3 -m pip install mamba-ssm --no-cache-dir --no-build-isolation
# Specify a new user (USER_NAME and USER_UID are specified via --build-arg)
ARG USER_UID
ARG USER_NAME
ENV USER_GID=$USER_UID
ENV USER_GROUP="users"

# Create the user
RUN mkdir /home/$USER_NAME && \
    useradd -l -d /home/$USER_NAME -u $USER_UID -g $USER_GROUP $USER_NAME && \
    mkdir /home/$USER_NAME/.local && \
    chown -R ${USER_UID}:${USER_GID} /home/$USER_NAME/

CMD ["/bin/bash"]