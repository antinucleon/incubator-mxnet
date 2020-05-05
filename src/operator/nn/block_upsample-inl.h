/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file upsampling-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_NN_BLOCK_UPSAMPLE_INL_H_
#define MXNET_OPERATOR_NN_BLOCK_UPSAMPLE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "./block_upsample_expr.h"

namespace mxnet {
namespace op {

namespace bup_enum {
enum UpSamplingOpInputs {kData, kWeight};
enum UpSamplingOpOutputs {kOut};
enum UpSamplingType {kNearest};
enum UpSamplingMultiInputMode {kConcat, kSum};
}  // namespace bup_enum

struct BlockUpSamplingParam : public dmlc::Parameter<BlockUpSamplingParam> {
  int scale_h;
  int scale_w;
  int num_filter;
  int sample_type;
  int num_args;
  int multi_input_mode;
  uint64_t workspace;
  DMLC_DECLARE_PARAMETER(BlockUpSamplingParam) {
    DMLC_DECLARE_FIELD(scale_h)
    .set_range(1, 1000)
    .describe("Up sampling scale");
    DMLC_DECLARE_FIELD(scale_w)
    .set_range(1, 1000)
    .describe("Up sampling scale");
    DMLC_DECLARE_FIELD(num_filter)
    .describe("Input filter. Only used by bilinear sample_type."
              "Since bilinear upsampling uses deconvolution, num_filters "
              "is set to the number of channels.")
    .set_default(0);
    DMLC_DECLARE_FIELD(sample_type)
    .add_enum("nearest", bup_enum::kNearest)
    .set_default(bup_enum::kNearest)
    .describe("upsampling method");
    DMLC_DECLARE_FIELD(multi_input_mode)
    .add_enum("concat", bup_enum::kConcat)
    .add_enum("sum", bup_enum::kSum)
    .set_default(bup_enum::kConcat)
    .describe("How to handle multiple input. concat means concatenate upsampled "
    "images along the channel dimension. sum means add all images together, "
    "only available for nearest neighbor upsampling.");
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs to be upsampled. For nearest neighbor "
    "upsampling, this can be 1-N; the size of output will be"
    "(scale*h_0,scale*w_0) and all other inputs will be upsampled to the"
    "same size. For bilinear upsampling this must be 2; 1 input and 1 weight.");
    DMLC_DECLARE_FIELD(workspace).set_default(512).set_range(0, 8192)
    .describe("Tmp workspace for deconvolution (MB)");
  }
};  // struct BlockUpSamplingParam

template<typename xpu, typename DType>
void BlockUpSamplingForward(const OpContext &ctx, const BlockUpSamplingParam &param,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(in_data.size(), static_cast<size_t>(param.num_args));
  CHECK_EQ(out_data.size(), 1U);
  if (req[bup_enum::kOut] == kNullOp) {
    return;
  }
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 4, DType> out = out_data[bup_enum::kOut].get<xpu, 4, DType>(s);
  if (param.num_args > 1) {
    int begin = 0;
    for (int i = 0; i < param.num_args; ++i) {
      Tensor<xpu, 4, DType> data = in_data[i].get<xpu, 4, DType>(s);
      int end = begin + data.size(1);
      int scale_h = out_data[bup_enum::kOut].size(2)/in_data[i].size(2);
      int scale_w = out_data[bup_enum::kOut].size(3)/in_data[i].size(3);
      if (param.multi_input_mode == bup_enum::kSum) {
        if (i == 0) {
          Assign(out, req[bup_enum::kOut], upsampling_block(data, scale_h, scale_w));
        } else {
          out += upsampling_block(data, scale_h, scale_w);
        }
      } else {
        Assign(slice<1>(out, begin, end), req[bup_enum::kOut], upsampling_block(data, scale_h, scale_w));
      }
      begin = end;
    }
  } else {
    Tensor<xpu, 4, DType> data = in_data[bup_enum::kData].get<xpu, 4, DType>(s);
    Assign(out, req[bup_enum::kOut], upsampling_block(data, param.scale_h, param.scale_w));
  }
}

template<typename xpu, typename DType>
void BlockUpSamplingBackward(const OpContext &ctx, const BlockUpSamplingParam &param,
                        const TBlob &out_grad, const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad) {
  using namespace mshadow;
  using namespace mshadow::expr;
  LOG(FATAL) << "Lazy";
}


template<typename xpu>
void BlockUpSamplingCompute(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx, const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  const BlockUpSamplingParam& param = nnvm::get<BlockUpSamplingParam>(attrs.parsed);
  if (param.sample_type == bup_enum::kNearest) {
    MSHADOW_REAL_TYPE_SWITCH(inputs[bup_enum::kData].type_flag_, DType, {
      BlockUpSamplingForward<xpu, DType>(ctx, param, inputs, req, outputs);
    });
  } else {
    LOG(FATAL) << "Unknown sample type";
  }
}



}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_UPSAMPLING_INL_H_
