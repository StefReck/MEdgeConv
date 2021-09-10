# tf dev docker image; see https://hub.docker.com/r/tensorflow/tensorflow/tags/
DOCKER_IMAGE="tensorflow/tensorflow:2.6.0-gpu"

# fix the 'cuda_fp16.h: No such file or directory' error:
FIX_CMND="mkdir -p /usr/local/lib/python3.6/dist-packages/tensorflow/include/third_party/gpus/cuda/include && cp -r /usr/local/cuda/targets/x86_64-linux/include/* /usr/local/lib/python3.6/dist-packages/tensorflow/include/third_party/gpus/cuda/include"

docker run -v "$PWD":/tf_ops $DOCKER_IMAGE /bin/bash -c "${FIX_CMND} && cd tf_ops && make clean && make"
