FROM tensorflow/tensorflow:2.4.0-gpu
WORKDIR /root

RUN python -m pip install --upgrade pip

RUN pip install OpenNMT-tf

RUN apt-key del A4B469963BF863CC || true

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y build-essential \
    zsh \
    vim \
    wget \
    cmake \
    git

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

WORKDIR /home

RUN git clone https://github.com/google/sentencepiece.git && \
    cd sentencepiece && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

RUN which spm_train

SHELL ["/bin/zsh", "-c"]

WORKDIR /workspace

ENTRYPOINT ["zsh"]

