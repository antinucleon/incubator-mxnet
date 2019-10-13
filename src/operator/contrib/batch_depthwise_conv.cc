#include "./batch_depthwise-inl.h"
#include "../elemwise_op_common.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(BatchDWParam);


static inline index_t AddPad(index_t dsize, index_t pad) {
  return dsize + 2 * pad;
}

static inline std::vector<std::string> ListArguments(const BatchDWParam& param_) {
  return {"data", "weight"};
}


static bool BatchDWShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector *in_shape,
                             mxnet::ShapeVector *out_shape) {
  using namespace mshadow;
  const BatchDWParam& param_ = nnvm::get<BatchDWParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  out_shape->resize(1, mxnet::TShape());
  const mxnet::TShape &dshp = (*in_shape)[bdw::kData];
  if (!mxnet::ndim_is_known(dshp)) return false;
  // 2d conv
  CHECK_EQ(dshp.ndim(), 4U) \
    << "Input data should be 4D in batch-num_filter-y-x";
  Shape<4> dshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNCHW);
  Shape<4> wshape = Shape4(dshape[0], param_.channels,
        param_.kernel_size, param_.kernel_size);
  SHAPE_ASSIGN_CHECK(*in_shape, bdw::kWeight, wshape);
  CHECK_EQ(dshape[1], param_.channels) << "input num_filter must equal group size";

  CHECK_GT(param_.kernel_size, 0U) \
    << "incorrect kernel size: " << param_.kernel_size;
  CHECK_GT(param_.stride, 0U) \
    << "incorrect stride size: " << param_.stride;
  Shape<4> oshape;
  oshape[0] = dshape[0];
  oshape[1] = param_.channels;
  oshape[2] = dshape[2] != -1 ?
      (AddPad(dshape[2], param_.pad) - 1) / param_.stride + 1 : -1;
  oshape[3] = dshape[3] != -1 ?
      (AddPad(dshape[3], param_.pad) - 1) / param_.stride + 1 : -1;
  SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));
  // Perform incomplete shape inference. Fill in the missing values in data shape.
  // 1) We can always fill in the batch_size.
  // 2) We can back-calculate the input height/width if the corresponding stride is 1.
  oshape = ConvertLayout((*out_shape)[0].get<4>(), param_.layout.value(), kNCHW);
  dshape[0] = oshape[0];
  if (oshape[2] != -1 && param_.stride == 1) {
    dshape[2] = oshape[2] - 2 * param_.pad;
  }
  if (oshape[3] != -1 && param_.stride == 1) {
    dshape[3] = oshape[3] - 2 * param_.pad;
  }
  SHAPE_ASSIGN_CHECK(*in_shape, bdw::kData,
      ConvertLayout(dshape, kNCHW, param_.layout.value()));
  // Check whether the kernel sizes are valid
    return true;
}

static bool BatchDWType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_type, std::vector<int> *out_type) {
  const BatchDWParam& param_ = nnvm::get<BatchDWParam>(attrs.parsed);
  CHECK_GE(in_type->size(), 1U);
  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  for (size_t i = 0; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments(param_)[i]);
    }
  }
  out_type->clear();
  out_type->push_back(dtype);
  return true;
}



void BatchDWParamParser(nnvm::NodeAttrs* attrs) {
  using namespace mshadow;
  BatchDWParam param_;
  try {
    param_.Init(attrs->dict);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCHW;
  attrs->parsed = std::move(param_);
}


struct BatchDWGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    heads.push_back(n->inputs[bdw::kData]);
    heads.push_back(n->inputs[bdw::kWeight]);
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

NNVM_REGISTER_OP(BatchDW)
.set_num_inputs([](const NodeAttrs& attrs) {
  return 2;
})
.set_num_outputs(1)
.set_attr_parser(BatchDWParamParser)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "weight"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", BatchDWShape)
.set_attr<nnvm::FInferType>("FInferType", BatchDWType)
.set_attr<FCompute>("FCompute<cpu>", BatchDWCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", BatchDWGrad{"_backward_BatchDW"})
.add_argument("data", "NDArray-or-Symbol", "Input data to the ConvolutionOp.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_arguments(BatchDWParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_BatchDW)
.set_num_outputs([](const NodeAttrs& attrs) {
  return 2;
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(BatchDWParamParser)
.set_attr<FCompute>("FCompute<cpu>", BatchDWGradCompute<cpu>);


}  // namespace op
} // namespace mxnet