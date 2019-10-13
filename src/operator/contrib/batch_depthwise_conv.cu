#include "./batch_depthwise-inl.h"
#include <vector>


namespace mxnet {
namespace op {

template<>
void BatchDWCompute<gpu>(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  const BatchDWParam& param = nnvm::get<BatchDWParam>(attrs.parsed);
  mxnet::ShapeVector in_shape(inputs.size());
  mxnet::ShapeVector out_shape(1, outputs[0].shape_);
  for (size_t i = 0; i < in_shape.size(); i++)
    in_shape[i] = inputs[i].shape_;
  int dtype = inputs[bdw::kData].type_flag_;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    BatchDWOp<gpu, DType> op;
    op.Init(param, in_shape, out_shape);
    op.Forward(ctx, inputs, req, outputs);
  })
}

template<>
void BatchDWGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  const BatchDWParam& param = nnvm::get<BatchDWParam>(attrs.parsed);
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  const TBlob &out_grad = inputs[0];
  const std::vector<TBlob> &in_grad = outputs;
  int dtype = out_grad.type_flag_;
  mxnet::ShapeVector in_shape(in_data.size());
  mxnet::ShapeVector out_shape(1, out_grad.shape_);
  for (size_t i = 0; i < in_shape.size(); i++)
    in_shape[i] = in_data[i].shape_;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    BatchDWOp<gpu, DType> op;
    op.Init(param, in_shape, out_shape);
    op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
  })
}


NNVM_REGISTER_OP(BatchDW)
.set_attr<FCompute>("FCompute<gpu>", BatchDWCompute<gpu>);
  
NNVM_REGISTER_OP(_backward_BatchDW)
.set_attr<FCompute>("FCompute<gpu>", BatchDWGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet