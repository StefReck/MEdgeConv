#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("KnnGraph")
    .Input("x: T")
    .Input("ptr_x: int32")
    .Attr("k: int")
    .Attr("T: {float, int32} = DT_FLOAT")
    .Output("col: int32")
    .Output("dist: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input_x;  // shape (n_nodes, dims)
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_x));

        shape_inference::ShapeHandle input_ptr_x;  // shape (batchsize + 1, )
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input_ptr_x));

        int k;
        TF_RETURN_IF_ERROR(c->GetAttr("k", &k));
        shape_inference::ShapeHandle output_shape = c->MakeShape({c->Dim(c->input(0), 0), k});
        c->set_output(0, output_shape);
        c->set_output(1, output_shape);
        return Status::OK();
    });
