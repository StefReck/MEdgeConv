# make op to compile everything, make gpu_only to only compile the gpu kernel
CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

SRCS = cc/kernels/knn_graph_kernels.cc $(wildcard cc/kernels/*.h) $(wildcard cc/ops/*.cc) $(wildcard cc/ops/*.cu)
GPU_ONLY_TARGET_LIB = python/ops/_knn_graph_ops.cu.o
TARGET_LIB = python/ops/_knn_graph_ops.so


TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}


op: $(TARGET_LIB)
$(TARGET_LIB): $(SRCS) $(GPU_ONLY_TARGET_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart

gpu_only: $(GPU_ONLY_TARGET_LIB)
$(GPU_ONLY_TARGET_LIB): cc/kernels/knn_graph_kernels.cu
	$(NVCC) -std=c++11 -c -o $@ $^  $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

clean:
	rm -f $(GPU_ONLY_TARGET_LIB) $(TARGET_LIB)
