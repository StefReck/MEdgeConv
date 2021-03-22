#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "knn_graph.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// CPU specialization of actual computation.
template <typename T>
struct KnnGraphFunctor<CPUDevice, T> {
    void operator()(
        const CPUDevice& d,
        const T* x_flat,
        const int* ptr_x_flat,
        float* dist_flat,
        int* col_flat,
        int K,
        int dim,
        const int batchsize)
    {
        for (int batch_idx = 0; batch_idx<batchsize; batch_idx++)
        {
            // node idx where this batch starts
            int y_start_idx = ptr_x_flat[batch_idx];
            // node idx where this batch ends
            int y_end_idx = ptr_x_flat[batch_idx + 1];

            for (int64_t n_y = y_start_idx; n_y < y_end_idx; n_y += 1) {

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
        }
    }
};

// TODO check that diff(x_ptr) >= k in kernels (segfault otherwise)
template <typename Device, typename T>
class KnnGraphOp : public OpKernel {
    public:
        explicit KnnGraphOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
            OP_REQUIRES(context, k_ >= 1, errors::InvalidArgument("Need k >= 1, got ", k_));
        }

        void Compute(OpKernelContext* context) override {
            // nodes; shape (n_nodes, dim)
            const Tensor& x = context->input(0);
            // node indices where batches start; shape (batchsize+1, )
            const Tensor& ptr_x = context->input(1);

            OP_REQUIRES(context, TensorShapeUtils::IsMatrix(x.shape()),
                errors::InvalidArgument("x must be a 2-D tensor."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(ptr_x.shape()),
                errors::InvalidArgument("ptr_x must be a 1-D tensor."));

            int64_t n_nodes = x.shape().dim_size(0);
            int dim = x.shape().dim_size(1);
            auto flat_shape = TensorShape();
            flat_shape.AddDim(n_nodes);
            flat_shape.AddDim(k_);
            const int batchsize = ptr_x.shape().dim_size(0)-1;

            // Create output tensors
            Tensor* col = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, flat_shape, &col));  // shape (n_nodes, k_ )

            Tensor* dist_p = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(1, flat_shape, &dist_p));

            KnnGraphFunctor<Device, T>()(
                context->eigen_device<Device>(),
                x.flat<T>().data(),
                ptr_x.flat<int>().data(),
                dist_p->flat<float>().data(),
                col->flat<int>().data(),
                k_,
                dim,
                batchsize
            );
        }

    private:
        int k_;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("KnnGraph").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      KnnGraphOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  extern template struct KnnGraphFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("KnnGraph").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      KnnGraphOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
}
}  // namespace tensorflow
