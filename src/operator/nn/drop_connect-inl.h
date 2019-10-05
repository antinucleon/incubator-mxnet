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
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../random/sampler.h"
#include "../tensor/elemwise_binary_broadcast_op.h"


namespace drop_connect {
enum DropConnectOpInputs {kData};
enum DropConnectOpOutputs {kOut, kMask};
enum DropConnectOpForwardResource {kRandom};
enum DropConnectOpMode {kTraining, kAlways};
}  // namespace drop_connect

namespace mxnet {
namespace op {

const int MAX_DIM = 5;

struct DropConnectParam : public dmlc::Parameter<DropConnectParam> {
  float p;
  int mode;
  int warm_up_step;
  mxnet::TShape axes;
  dmlc::optional<bool> cudnn_off;
  DMLC_DECLARE_PARAMETER(DropConnectParam) {
    DMLC_DECLARE_FIELD(p).set_default(0.5)
    .set_range(0, 1)
    .describe("Fraction of the input that gets dropped out during training time.");
    DMLC_DECLARE_FIELD(warm_up_step).set_default(0)
    .describe("Fraction of the input that gets dropped out during training time.");
    DMLC_DECLARE_FIELD(mode)
    .add_enum("training", drop_connect::kTraining)
    .add_enum("always", drop_connect::kAlways)
    .set_default(drop_connect::kTraining)
    .describe("Whether to only turn on drop_connect during training or to also turn on for inference.");
    DMLC_DECLARE_FIELD(axes).set_default(mxnet::TShape(0, 0))
    .describe("Axes for variational drop_connect kernel.");
    DMLC_DECLARE_FIELD(cudnn_off).set_default(dmlc::optional<bool>(false))
    .describe("Whether to turn off cudnn in drop_connect operator. "
              "This option is ignored if axes is specified.");
  }
};  // struct DropConnectParam

template<typename xpu, typename DType>
class DropConnectOp {
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
                                    DType *mask_out,
                                    const real_t pkeep) {
      RNG_KERNEL_LOOP(xpu, DType, id, gen, N, step, {
        const real_t rand_num = static_cast<real_t>(genImpl.uniform());
        mask_out[i] = mshadow_op::floor::Map<real_t>(rand_num + pkeep) * (1.0f / pkeep);
      });
    }
  };


  explicit DropConnectOp(const DropConnectParam &param, Context ctx) {
    this->pkeep_ = 1.0f - param.p;
    this->mode_ = static_cast<drop_connect::DropConnectOpMode>(param.mode);
    this->axes_ = param.axes;
    this->drop_connect_passthrough_ = true;
  }

  ~DropConnectOp() {
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    this->drop_connect_passthrough_ = true;
    if (req[drop_connect::kOut] != kNullOp) {
      CHECK_EQ(in_data.size(), 1U);
      if (ctx.is_train) {
        CHECK_EQ(out_data.size(), 2U);
      }
      Stream<xpu> *s = ctx.get_stream<xpu>();
      const TBlob &in = in_data[drop_connect::kData];
      const int num_batch = in_data[drop_connect::kData].size(0);
      const TBlob &out = out_data[drop_connect::kOut];
      const TBlob &mask = out_data[drop_connect::kMask];
      if (this->pkeep_ < 1 && (ctx.is_train || this->mode_ == drop_connect::kAlways)) {
        this->drop_connect_passthrough_ = false;
        if (this->axes_.ndim() == 0) {
          RandGenerator<xpu, DType> *pgen = ctx.requested[0].get_parallel_random<xpu, DType>();
          CHECK_NOTNULL(pgen);
          CHECK(req[drop_connect::kOut] != kAddTo);

          // initialize the mask
          LaunchRNG<BernoulliKernel, xpu>(s, pgen, 
			                                    mask.Size(),
                                          mask.dptr<DType>(),
                                          this->pkeep_);
          mxnet::TShape new_lshape, new_rshape, new_oshape;
          int ndim = BinaryBroadcastShapeCompact(in.shape_, mask.shape_, out.shape_,
                                         &new_lshape, &new_rshape, &new_oshape);
            BROADCAST_NDIM_SWITCH(ndim, NDim, {
              mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
              mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
              mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
              mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, DType, op::mshadow_op::mul>, xpu>::
              template LaunchEx(s, new_oshape.Size(), req[0], lstride, rstride, oshape,
              in.dptr<DType>(), mask.dptr<DType>(), out.dptr<DType>());
            });
	        return;
        }
      } else {
        MXNET_ASSIGN_REQ_SWITCH(req[drop_connect::kOut], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
            s, out.Size(), out.dptr<DType>(), in.dptr<DType>());
        });
      }
    }
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &out_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (!this->drop_connect_passthrough_) {
      this->drop_connect_passthrough_ = true;
      const TBlob &gdata = in_grad[drop_connect::kData];
      const TBlob &grad = out_grad[drop_connect::kOut];
      const TBlob &mask = out_data[drop_connect::kMask];
      if (this->axes_.ndim() == 0) {
        // standard case for drop_connect
        mxnet::TShape new_lshape, new_rshape, new_oshape;
          int ndim = BinaryBroadcastShapeCompact(grad.shape_, mask.shape_, gdata.shape_,
                                         &new_lshape, &new_rshape, &new_oshape);
            BROADCAST_NDIM_SWITCH(ndim, NDim, {
              mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
              mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
              mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
              mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, DType, op::mshadow_op::mul>, xpu>::
              template LaunchEx(s, new_oshape.Size(), req[0], lstride, rstride, oshape,
              grad.dptr<DType>(), mask.dptr<DType>(), gdata.dptr<DType>());
            });
        //MXNET_ASSIGN_REQ_SWITCH(req[drop_connect::kData], Req, {
        //  mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::mul, Req>, xpu>::Launch(
        //    s, gdata.Size(), gdata.dptr<DType>(), grad.dptr<DType>(), mask.dptr<DType>());
        //});
        return;
      } 
    } else {
      const TBlob& gdata = in_grad[drop_connect::kData];
      const TBlob& grad = out_grad[drop_connect::kOut];
      MXNET_ASSIGN_REQ_SWITCH(req[drop_connect::kData], Req, {
        mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
          s, gdata.Size(), gdata.dptr<DType>(), grad.dptr<DType>());
      });
    }
  }

 private:
  /*! \brief DropConnect rate (keep when the generated random number is less than this value) */
  real_t pkeep_;
  /*! \brief DropConnect mode */
  drop_connect::DropConnectOpMode mode_;
  /*! \brief Axes on which drop_connect mask is shared in the form of broadcast multiply */
  mxnet::TShape axes_;
  /*! \brief Flag to record whether forward is executed in pass-through mode */
  bool drop_connect_passthrough_;
};  // class DropConnectOp

template<typename xpu>
void DropConnectCompute(const OpStatePtr& state,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    DropConnectOp<xpu, DType>& op = state.get_state<DropConnectOp<xpu, DType>>();
    op.Forward(ctx, inputs, req, outputs);
  });
}

template<typename xpu>
void DropConnectGradCompute(const OpStatePtr& state,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1);
  CHECK_EQ(req.size(), 1);
  std::vector<TBlob> out_grads(2);
  std::vector<TBlob> out_data(2);
  out_grads[drop_connect::kOut] = inputs[0];
  out_data[drop_connect::kMask] = inputs[1];

  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    DropConnectOp<xpu, DType>& op = state.get_state<DropConnectOp<xpu, DType>>();
    op.Backward(ctx, out_grads, out_data, req, outputs);
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_DROPCONNECT_INL_H_
