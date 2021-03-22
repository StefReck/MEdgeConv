#ifndef KNN_GRAPH_H_
#define KNN_GRAPH_H_

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct KnnGraphFunctor {
  void operator()(
        const Device& d,
        const T* x_flat,
        const int* ptr_x_flat,
        float* dist_flat,
        int* col_flat,
        int K,
        int dim,
        const int batchsize);
};

}  // namespace functor
}  // namespace tensorflow

#endif //KNN_GRAPH_H_
