#ifndef MXNET_OPERATOR_NN_CONTRIB_BATCH_DEPTHWISE_INL_H_
#define MXNET_OPERATOR_NN_CONTRIB_BATCH_DEPTHWISE_INL_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"



#if MXNET_USE_CUDA

#include "./batch_depthwise.cuh"
#include "../../common/cuda_utils.h"



namespace mxnet {
namespace op {

namespace bdw {
enum BatchDWOpInputs {kData, kWeight};
enum BatchDWOpOutputs {kOut};
}

struct BatchDWParam : public dmlc::Parameter<BatchDWParam> {
  int kernel_size;
  int stride;
  int pad;
  int channels;
  dmlc::optional<int> layout;
  DMLC_DECLARE_PARAMETER(BatchDWParam) {
    DMLC_DECLARE_FIELD(kernel_size);
    DMLC_DECLARE_FIELD(stride).set_default(0);
    DMLC_DECLARE_FIELD(pad).set_default(0);
    DMLC_DECLARE_FIELD(channels);
    DMLC_DECLARE_FIELD(layout)
    .add_enum("NCHW", mshadow::kNCHW)
    .set_default(dmlc::optional<int>());
  }

  bool operator==(const BatchDWParam& other) const {
    return this->kernel_size == other.kernel_size &&
           this->stride == other.stride &&
           this->pad == other.pad &&
           this->channels == other.channels &&
           this->layout == other.layout;
  }

};

void BatchDWParamParser(nnvm::NodeAttrs* attrs);

typedef ParamOpSign<BatchDWParam> BatchDWSignature;

}  // namespace op
}  // namespace mxnet


namespace std {
template<>
struct hash<mxnet::op::BatchDWParam> {
  size_t operator()(const mxnet::op::BatchDWParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.kernel_size);
    ret = dmlc::HashCombine(ret, val.stride);
    ret = dmlc::HashCombine(ret, val.pad);
    ret = dmlc::HashCombine(ret, val.channels);
    ret = dmlc::HashCombine(ret, val.layout);
    return ret;
  }
};
}  // namespace std


namespace mxnet {
namespace op {

using namespace tf::depthwise_conv;

template<typename xpu, typename DType>
class BatchDWOp {
 public:
  void Init(BatchDWParam p) {
    args_.in_channel = p.channels;
    args_.filter_height = p.kernel_size;
    args_.filter_width = p.kernel_size;
    args_.stride_height = p.stride;
    args_.stride_width = p.stride;
    args_.pad_height = p.pad;
    args_.pad_width = p.pad;
    args_.out_channel =  p.channels;
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // CHECK_EQ(req[bdw::kOut], kWriteTo);
    // args_.batch = in_data[bdw::kData].shape_[0];
    // args_.in_height = in_data[bdw::kData].shape_[2];
    // args_.in_width = in_data[bdw::kData].shape_[3];
    // args_.out_height = out_data[bdw::kOut].shape_[2];
    // args_.out_width = out_data[bdw::kOut].shape_[3];


  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob>& out_grad,
                const std::vector<TBlob>& in_data,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // CHECK_EQ(out_grad.size(), 1U);
    // args_.batch = in_data[bdw::kData].shape_[0];
    // args_.in_height = in_data[bdw::kData].shape_[2];
    // args_.in_width = in_data[bdw::kData].shape_[3];
    // args_.out_height = out_grad[bdw::kOut].shape_[2];
    // args_.out_width = out_grad[bdw::kOut].shape_[3];

  }

 private:
  DepthwiseArgs args_;

};  // BatchDWop


template<typename xpu>
void BatchDWCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx, const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  const BatchDWParam& param = nnvm::get<BatchDWParam>(attrs.parsed);
  MSHADOW_REAL_TYPE_SWITCH(inputs[bdw::kData].type_flag_, DType, {
    BatchDWOp<xpu, DType> op;
    op.Init(param);
    op.Forward(ctx, inputs, req, outputs);
  });
}

template<typename xpu>
void BatchDWGradCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx, const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  const BatchDWParam& param = nnvm::get<BatchDWParam>(attrs.parsed);
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  const TBlob &out_grad = inputs[0];
  const std::vector<TBlob> &in_grad = outputs;

  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    BatchDWOp<xpu, DType> op;
    op.Init(param);
    op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
  });
}

}  // namespace op
}  // namespace mxnet
#endif   // MXNET_USE_CUDA
#endif // MXNET_OPERATOR_NN_CONTRIB_BATCH_DEPTHWISE_INL_H_