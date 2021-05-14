# tf dev docker image; see https://hub.docker.com/r/tensorflow/tensorflow/tags/?page=1&ordering=last_updated&name=custom-op-gpu
DOCKER_IMAGE="tensorflow/tensorflow:2.5.0-custom-op-gpu-ubuntu16"

# fix the 'cuda_fp16.h: No such file or directory' error:
FIX_CMND="mkdir -p /usr/local/lib/python3.6/dist-packages/tensorflow/include/third_party/gpus/cuda/include && cp -r /usr/local/cuda/targets/x86_64-linux/include/* /usr/local/lib/python3.6/dist-packages/tensorflow/include/third_party/gpus/cuda/include"

docker run $DOCKER_IMAGE /bin/bash -v "$PWD":/tf_ops -c "${FIX_CMND} && cd tf_ops && make clean && make"
