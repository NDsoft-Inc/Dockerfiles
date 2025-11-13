# Train onmt-tf model in a docker container

## Notice
- Last update: 11.10(Mon) by Dongik - rewrite a dockerfile, now you can just run a container with the `train/opennmt-tf:2.4.0-gpu` image without installing any dependency and sentencepiece model.
- Update: 10.10(Fri) by Dongik - feature with building a base image.

## Fast execution

- Run below command, if the image is already exists. Container name rule (e.g. user_onmt_gpu-1_2)

    `docker run --rm -it --name <container_name> --runtime=nvidia --gpus '"device=<gpu_id>"' -v <host_mount_path>:/workspace train/opennmt-tf:2.4.0-gpu`

    `(e.g.) docker run -it --name sdi_onmt_gpu-3 --runtime=nvidia --gpus '"device=3"' -v /home/sdi/onmt-tf/:/workspace train/opennmt-tf:2.4.0-gpu`

- Get into the container.

    `docker exec -it <container_name> zsh`

    `(e.g.) docker exec -it sdi_onmt_gpu-3 zsh`

## Build opennmt Image

0) (Optional) If you are using docker-desktop in Ubuntu, you may have to remove following configuration. Ubuntu don't have "credential helper". (docker-credential-desktop이라는 Docker 인증 헬퍼를 못 찾아서 발생. Mac이나 Windows용 Docker Desktop에서는 기본적으로 이 헬퍼를 써서 Docker Hub에 로그인하는데, Ubuntu 환경에서는 이게 없어서 인증에 실패)

    ```
    vi ~/.docker/config.json

    # remove "credsStore" variable
    {
        "credsStore": "desktop",
    }

    docker builder prune
    ```

1) To build docker iamge, run following command

    `docker build -t train/opennmt-tf:2.4.0-gpu -f Dockerfile .`

2) To make container, run following command

    `docker run --rm -it --name <my_container> --runtime=nvidia --gpus '"device=<gpu_id>"' -v <host_mount_path>:/workspace train/opennmt-tf:2.4.0-gpu`

3) (Optional) For better interface, and in case `oh-my-zsh` is not installed, then install `oh-my-zsh`.

    `sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"`

4) (Optional) If GPG key dose not match and fail to build an image, run commands below in the container. 

    `apt update` doesn't work in the given docker image: seems to work now.

    ```
    apt update && apt install -y build-essential \
    zsh \
    vim \
    wget \
    cmake \
    git
    ```

5) (Optional) If you fail to make sentence piece model in the container, build and install SentencePiece command line tools from C++ source. (https://github.com/google/sentencepiece)

    ```
    cd /home
    git clone https://github.com/google/sentencepiece.git
    cd sentencepiece
    mkdir build && cd build
    cmake ..
    make -j
    make install
    ldconfig

    # verification
    which spm_train
    ```