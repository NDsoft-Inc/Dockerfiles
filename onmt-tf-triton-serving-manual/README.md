# ONMT-tf Inference by Triton Inference Server

## Notice
- Last update: 11.11(Tue) 2025 by Dongik - Update dockerfile, now you can just run an image to serve the model.
- Update: 10.10(Fri) 2025 by Dongik.

## About Triton
$NVIDIA\;Triton^{TM}$ Inference Server 는 모델 배포 및 실행을 표준화할 수 있도록 도와주고, 프로덕션 환경에 빠르고 확장 가능한 AI를 제공하는 오픈 소스 추론 지원 소프트웨어. It supports various framework suchas TF, Pytorch, onnx and TensorRT.

## Requirements

- ONMT-tf model parameters
- Tokenizer (sentencpiece model)
- Docker image: `nvcr.io/nvidia/tritonserver:23.10-py3`

## Build a docker image
Run command below to make an image.
```
docker build -t deploy/opennmt-tf/tritonserver:23.10-py3 .
```

## Run a container to serve and test ONMT-tf
Run commands to serve a model.

```
docker run --rm -it --gpus "device=4" --name <container_name> --shm-size=256m -v <host_mount_path_to_params>:/models deploy/opennmt-tf/tritonserver:23.10-py3

(e.g.) docker run --rm -it --gpus "device=4" --name sdi_deploy_gpu-4 --shm-size=256m -v /home/sdi/onmt-tf/komy/triton/:/models deploy/opennmt-tf/tritonserver:23.10-py3
```

If you see the status below, the model is ready to respond

```
+---------------+---------+--------+  
| Model         | Version | Status |  
+---------------+---------+--------+  
| koen          | 1       | READY  |  
| koja          | 1       | READY  |  
+---------------+---------+--------+
```

To test the model, you can run a script. Open additional terminal

```
docker exec -it <container_name> zsh

python3 triton-test.py
```

Then, you can input a sentencepiece directory.

```
==================================================
🚀 SentencePiece Translator 설정
==================================================
Triton 모델 이름: komy
SentencePiece 모델 파일 경로 (.model): /models/komy/wmt.komy.0516.model
```

## (Appendix) Docker option configuration

Start docker container by running following command.
```
docker run --rm -it --gpus "device=4" --name sdi_triton --shm-size=256m -v /home/sdi/onmt-tf/komy/triton:/models nvcr.io/nvidia/tritonserver:23.10-py3 

# if you build your own image run this
(optional) docker run --rm -it --gpus "device=4" --name sdi_triton --shm-size=256m -v /home/sdi/onmt-tf/komy/triton:/models my/tritonserver:23.10-py3 
```
- --rm: 컨테이너를 실행하고 프로세스가 종료되면 즉시 삭제
- --it: 표준 입력(stdin) 터미널에서 입력을 컨테이너로 전달, 가상 터미널(tty) 을 할당, 쉘 환경처럼 표시
- --gpus: 컨테이너에 GPU 자원을 할당 (e.g. --gpus "device=0,2")
- --shm-size: 컨테이너의 공유 메모리(shared memory) 크기를 지정
    - 리눅스에서는 /dev/shm 경로가 공유 메모리 공간
    - Docker 컨테이너는 기본적으로 /dev/shm 크기가 64MB로 제한
    - 대규모 연산(예: PyTorch DataLoader 병렬 로딩, Chrome headless, multiprocessing 등)을 사용할 때 이 공간이 너무 작으면 오류 발생
- -v: <host_path>:<container_path>[:option]
    - ro: read only
    - rw (default): read and write
    - multiple mount support by calling multiple `-v` options (e.g.) 
        ```
        docker run -it --rm \
            -v $(pwd):/workspace \
            python:3.12 \
            python /workspace/train.py
        ```

Install following packages.    
```
pip install tritonclient[http]
pip install sentencepiece
```