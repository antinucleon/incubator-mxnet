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
 * \file drop_connect-inl.h
 * \brief
 * \author Bing Xu, Da Zheng, Hang Zhang
 */

#ifndef MXNET_OPERATOR_NN_DROPCONNECT_INL_H_
#define MXNET_OPERATOR_NN_DROPCONNECT_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../random/sampler.h"
#include "../tensor/elemwise_binary_broadcast_op.h"

namespace drop_res {
enum DropResOpInputs {
  kData,
};
enum DropResOpOutputs { kMask1, kMask2 };
enum DropResOpForwardResource { kRandom };
enum DropResOpMode { kTraining, kAlways };
}  // namespace drop_res

namespace mxnet {
namespace op {

const int MAX_DIM = 5;

struct DropResParam : public dmlc::Parameter<DropResParam> {
  float p;
  int mode;
  int warm_up_step;
  mxnet::TShape axes;
  dmlc::optional<bool> cudnn_off;
  DMLC_DECLARE_PARAMETER(DropResParam) {
    DMLC_DECLARE_FIELD(p).set_default(0.5).set_range(0, 1).describe(
        "Fraction of the input that gets dropped out during training time.");
    DMLC_DECLARE_FIELD(warm_up_step)
        .set_default(0)
        .describe(
            "Fraction of the input that gets dropped out during training "
            "time.");
    DMLC_DECLARE_FIELD(mode)
        .add_enum("training", drop_res::kTraining)
        .add_enum("always", drop_res::kAlways)
        .set_default(drop_res::kTraining)
        .describe(
            "Whether to only turn on drop_connect during training or to also "
            "turn on for inference.");
    DMLC_DECLARE_FIELD(axes)
        .set_default(mxnet::TShape(0, 0))
        .describe("Axes for variational drop_connect kernel.");
    DMLC_DECLARE_FIELD(cudnn_off)
        .set_default(dmlc::optional<bool>(false))
        .describe(
            "Whether to turn off cudnn in drop_connect operator. "
            "This option is ignored if axes is specified.");
  }
};  // struct DropResParam

template <typename xpu, typename DType>
class DropResOp {
 public:
  /*!
   * \brief DropConnect kernel, compute drop_connect tensor
   */

  struct BernoulliKernel {
    /*! \brief Bernoulli kernel for generating mask */
    MSHADOW_XINLINE static void Map(int id,
                                    RandGenerator<xpu, DType> gen,
                                    const int N,
                                    const int step,
                                    DType* mask1_out,
                                    DType* mask2_out,
                                    const real_t pkeep) {
      RNG_KERNEL_LOOP(xpu, DType, id, gen, N, step, {
        const real_t rand_num = static_cast<real_t>(genImpl.uniform());
        real_t value = mshadow_op::floor::Map<real_t>(rand_num + pkeep);
        mask1_out[i] = value;
        mask2_out[i] = 1.0f - value;
      });
    }
  };



  explicit DropResOp(const DropResParam& param, Context ctx) {
    this->pkeep_ = 1.0f - param.p;
    this->mode_ = static_cast<drop_res::DropResOpMode>(param.mode);
    this->axes_ = param.axes;
    this->drop_connect_passthrough_ = true;
  }

  ~DropResOp() {}

  void Forward(const OpContext& ctx,
               const std::vector<TBlob>& in_data,
               const std::vector<OpReqType>& req,
               const std::vector<TBlob>& out_data) {
    this->drop_connect_passthrough_ = true;
    if (req[drop_res::kMask1] != kNullOp) {
      CHECK_EQ(in_data.size(), 1U);
      if (ctx.is_train) {
        CHECK_EQ(out_data.size(), 2U);
      }
      Stream<xpu>* s = ctx.get_stream<xpu>();
      const TBlob& in = in_data[drop_res::kData];
      const int num_batch = in_data[drop_res::kData].size(0);
      const TBlob& mask_1 = out_data[drop_res::kMask1];
      const TBlob& mask_2 = out_data[drop_res::kMask2];
      if (this->pkeep_ < 1 &&
          (ctx.is_train || this->mode_ == drop_res::kAlways)) {
        this->drop_connect_passthrough_ = false;
        if (this->axes_.ndim() == 0) {
          RandGenerator<xpu, DType>* pgen =
              ctx.requested[0].get_parallel_random<xpu, DType>();
          CHECK_NOTNULL(pgen);
          CHECK(req[drop_res::kMask1] != kAddTo);

          // initialize the mask
          LaunchRNG<BernoulliKernel, xpu>(s, pgen, mask_1.Size(),
                                          mask_1.dptr<DType>(),
                                          mask_2.dptr<DType>(), this->pkeep_);
          return;
        }
      } else {
        MXNET_ASSIGN_REQ_SWITCH(req[drop_res::kMask1], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::assign_1, Req>, xpu>::Launch(
              s, mask_1.Size(), mask_1.dptr<DType>(), mask_1.dptr<DType>());
        });
        MXNET_ASSIGN_REQ_SWITCH(req[drop_res::kMask2], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::assign_0, Req>, xpu>::Launch(
              s, mask_2.Size(), mask_2.dptr<DType>(), mask_2.dptr<DType>());
        });
      }
    }
  }

  void Backward(const OpContext& ctx,
                const std::vector<TBlob>& out_grad,
                const std::vector<TBlob>& out_data,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // Stream<xpu>* s = ctx.get_stream<xpu>();

    // const TBlob& gdata = in_grad[drop_res::kData];

    // MXNET_ASSIGN_REQ_SWITCH(req[drop_res::kData], Req, {
    //   mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>,
    //                    xpu>::Launch(s, gdata.Size(), gdata.dptr<DType>(),
    //                                 gdata.dptr<DType>());
    // });
  }

 private:
  /*! \brief DropConnect rate (keep when the generated random number is less
   * than this value) */
  real_t pkeep_;
  /*! \brief DropConnect mode */
  drop_res::DropResOpMode mode_;
  /*! \brief Axes on which drop_connect mask is shared in the form of broadcast
   * multiply */
  mxnet::TShape axes_;
  /*! \brief Flag to record whether forward is executed in pass-through mode */
  bool drop_connect_passthrough_;
};  // class DropResOp

template <typename xpu>
void DropResCompute(const OpStatePtr& state,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    DropResOp<xpu, DType>& op = state.get_state<DropResOp<xpu, DType>>();
    op.Forward(ctx, inputs, req, outputs);
  });
}

template <typename xpu>
void DropResGradCompute(const OpStatePtr& state,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  // CHECK_EQ(inputs.size(), 2U);
  // CHECK_EQ(outputs.size(), 1);
  // CHECK_EQ(req.size(), 1);
  // std::vector<TBlob> out_grads(2);
  // std::vector<TBlob> out_data(2);
  // out_grads[drop_res::kOut] = inputs[0];
  // out_data[drop_res::kMask] = inputs[1];

  // MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
  //   DropResOp<xpu, DType>& op = state.get_state<DropResOp<xpu, DType>>();
  //   op.Backward(ctx, out_grads, out_data, req, outputs);
  // });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_DROPCONNECT_INL_H_
