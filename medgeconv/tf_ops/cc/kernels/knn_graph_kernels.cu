#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#define THREADS 1024

#include "knn_graph.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void KnnGraphCudaKernel(
        const T* x_flat,
        const int* ptr_x_flat,
        float* dist_flat,
        int* col_flat,
        int K,
        int dim)
    {
        const int64_t batch_idx = blockIdx.x;

        const int64_t y_start_idx = ptr_x_flat[batch_idx];
        const int64_t y_end_idx = ptr_x_flat[batch_idx + 1];

        for (int64_t n_y = y_start_idx + threadIdx.x; n_y < y_end_idx; n_y += THREADS) {

            // initialize distances
            for (int k_idx = 0; k_idx < K; k_idx++) {
                dist_flat[n_y * K + k_idx] = 1.0e38;
            }

            for (int64_t n_x = y_start_idx; n_x < y_end_idx; n_x++) {
                float tmp_dist = 0;

                for (int d = 0; d < dim; d++) {
                    tmp_dist += (x_flat[n_x * dim + d] - x_flat[n_y * dim + d]) *
                                (x_flat[n_x * dim + d] - x_flat[n_y * dim + d]);
                }

                for (int k_idx_1 = 0; k_idx_1 < K; k_idx_1++) {
                    if (dist_flat[n_y * K + k_idx_1] > tmp_dist) {
                        for (int k_idx_2 = K - 1; k_idx_2 > k_idx_1; k_idx_2--) {
                            dist_flat[n_y * K + k_idx_2] = dist_flat[n_y * K + k_idx_2 - 1];
                            col_flat[n_y * K + k_idx_2] = col_flat[n_y * K + k_idx_2 - 1];
                        }
                        dist_flat[n_y * K + k_idx_1] = tmp_dist;
                        col_flat[n_y * K + k_idx_1] = n_x;
                        break;
                    }
                }
            }
        }
    };

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct KnnGraphFunctor<GPUDevice, T> {
  void operator()(
    const GPUDevice& d,
    const T* x_flat,
    const int* ptr_x_flat,
    float* dist_flat,
    int* col_flat,
    int K,
    int dim,
    const int batchsize)
{
    int block_count = batchsize;
    KnnGraphCudaKernel<T><<<block_count, THREADS, 0, d.stream()>>>(
        x_flat, ptr_x_flat, dist_flat, col_flat, K, dim);
}
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct KnnGraphFunctor<GPUDevice, float>;
template struct KnnGraphFunctor<GPUDevice, int32>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
