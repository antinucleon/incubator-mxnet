#include "./batch_depthwise-inl.h"
#include "./batch_depthwise.cuh"
#include <vector>


namespace mxnet {
namespace op {


template<typename DType>
void BatchDW2dForwardGpu(mshadow::Stream<gpu> *stream,
                         const DepthwiseArgs& args,
                         const std::vector<TBlob> &in_data,
                         const std::vector<TBlob> &out_data) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace tf::depthwise_conv;
  using namespace tf::depthwise_conv::cuda;
  Tensor<gpu, 4, DType> data = in_data[bdw::kData].get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> weight = in_data[bdw::kWeight].get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> out = out_data[bdw::kOut].get<gpu, 4, DType>(stream);
  int num_output = out_data[bdw::kOut].shape_.Size();
  int block_num = std::min(num_output/mshadow::cuda::kBaseThreadNum + 1,
    mshadow::cuda::kMaxGridNum);
  auto s = mshadow::Stream<gpu>::GetStream(stream);
  DepthwiseConv2dForwardKernel<DType, -1, -1>
          <<<block_num, mshadow::cuda::kBaseThreadNum, 0, s>>>(data.dptr_,
                                                               weight.dptr_,
                                                               args,
                                                               num_output,
                                                               out.dptr_);
  MSHADOW_CUDA_POST_KERNEL_CHECK(DepthwiseConv2dForwardKernel);
}


template<typename DType>
void DepthwiseConv2dBackwardDataGpu(mshadow::Stream<gpu> *stream,
                                    const DepthwiseArgs& args,
                                    const std::vector<TBlob> &out_grad,
                                    const std::vector<TBlob> &in_data,
                                    const std::vector<TBlob> &in_grad) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace tf::depthwise_conv;
  using namespace tf::depthwise_conv::cuda;
  Tensor<gpu, 4, DType> out_g = out_grad[bdw::kOut].get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> weight = in_data[bdw::kWeight].get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> in_data_g = in_grad[bdw::kData].get<gpu, 4, DType>(stream);
  // select kernel

  int num_in_grad = in_grad[bdw::kData].shape_.Size();
  auto s = mshadow::Stream<gpu>::GetStream(stream);
  int block_num = std::min(num_in_grad/mshadow::cuda::kBaseThreadNum + 1,
                             mshadow::cuda::kMaxGridNum);
  DepthwiseConv2dBackwardDataKernel<DType>
        <<<block_num, mshadow::cuda::kBaseThreadNum, 0, s>>>(args,
                                                             out_g.dptr_,
                                                             weight.dptr_,
                                                             in_data_g.dptr_,
                                                             num_in_grad);
  MSHADOW_CUDA_POST_KERNEL_CHECK(DepthwiseConv2dBackwardDataKernel);
}

template<typename DType>
void DepthwiseConv2dBackwardFilterGpu(mshadow::Stream<gpu> *stream,
                                      const DepthwiseArgs& args,
                                      const std::vector<TBlob> &out_grad,
                                      const std::vector<TBlob> &in_data,
                                      const std::vector<TBlob> &in_grad) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace tf::depthwise_conv;
  using namespace tf::depthwise_conv::cuda;
  Tensor<gpu, 4, DType> out_g = out_grad[bdw::kOut].get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> in_d = in_data[bdw::kData].get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> weight_grad = in_grad[bdw::kWeight].get<gpu, 4, DType>(stream);
  // select kernel
    int num_out_grad = out_grad[conv::kOut].shape_.Size();
    auto s = mshadow::Stream<gpu>::GetStream(stream);
    int block_num = std::min(args.out_channel * args.batch, mshadow::cuda::kMaxGridNum);

    DepthwiseConv2dBackwardFilterKernel<DType, -1, -1>
          <<<block_num, mshadow::cuda::kBaseThreadNum, 0, s>>>(args,
                                                               out_g.dptr_,
                                                               in_d.dptr_,
                                                               weight_grad.dptr_,
                                                               num_out_grad);
    MSHADOW_CUDA_POST_KERNEL_CHECK(DepthwiseConv2dBackwardFilterKernel);
  }
}



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
    // op.Forward(ctx, inputs, req, outputs);
    auto stream = ctx.get_stream<gpu>();
    CHECK_EQ(req[bdw::kOut], kWriteTo);
    BatchDW2dForwardGpu<float>(stream, op.args_, inputs, outputs);
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
    // op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    auto stream = ctx.get_stream<gpu>();
    // data
    if (req[bdw::kData] != kNullOp) {
      if (req[bdw::kData] != kAddTo) {
        mshadow::Tensor<gpu, 4, DType> igrad = in_grad[bdw::kData].get<gpu, 4, DType>(stream);
        igrad = 0.0f;
      }
      DepthwiseConv2dBackwardDataGpu<float>(stream,
                                     op.args_,
                                     std::vector<TBlob>{out_grad},
                                     in_data, in_grad);

    }
    if (req[bdw::kWeight] != kNullOp) {
      if (req[bdw::kWeight] != kAddTo) {
        mshadow::Tensor<gpu, 4, DType> wgrad = in_grad[bdw::kWeight].get<gpu, 4, DType>(stream);
        wgrad = 0.0f;
      }
      DepthwiseConv2dBackwardFilterGpu<float>(stream,
        args_,
        out_grad,
        in_data,
        in_grad);
    }
  })
}


NNVM_REGISTER_OP(BatchDW)
.set_attr<FCompute>("FCompute<gpu>", BatchDWCompute<gpu>);
  
NNVM_REGISTER_OP(_backward_BatchDW)
.set_attr<FCompute>("FCompute<gpu>", BatchDWGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet