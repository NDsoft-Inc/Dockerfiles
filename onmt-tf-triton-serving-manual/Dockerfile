FROM nvcr.io/nvidia/tritonserver:23.10-py3

WORKDIR /workspace

RUN apt-get update && apt-get install -y zsh \
    && chsh -s /usr/bin/zsh

COPY *.py /workspace/

RUN pip install --no-cache-dir tritonclient[http] \
    && pip install --no-cache-dir sentencepiece

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

RUN echo 'set encoding=utf-8' >> /root/.vimrc && \
    echo 'set fileencodings=utf-8,cp949,euc-kr' >> /root/.vimrc && \
    echo 'set termencoding=utf-8' >> /root/.vimrc && \
    echo 'set fileencoding=utf-8' >> /root/.vimrc

SHELL ["/bin/zsh", "-c"]

WORKDIR /workspace

ENTRYPOINT ["tritonserver", "--model-repository=/models"]
