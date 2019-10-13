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
 * \file depthwise_convolution_tf.cuh
 * \brief some depthwise convolution CUDA kernel code. The main logic comes
 *        from tensorflow, but the filter's layerout and many argument names
 *        are different with origin version.
 * \author shuqian.qu@hobot.cc
*/
#ifndef MXNET_OPERATOR_NN_DEPTHWISE_CONVOLUTION_TF_CUH_
#define MXNET_OPERATOR_NN_DEPTHWISE_CONVOLUTION_TF_CUH_
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"
#include "./batch_depthwise.h"


namespace tf {
namespace depthwise_conv {



#define FULL_WARP_MASK 0xFFFFFFFF
#if CUDA_VERSION < 9000
template<typename DType>
__forceinline__ __device__ DType  __shfl_xor_sync(unsigned, DType val, int delta) {
  return __shfl_xor(val, delta);
}

template<typename DType>
__forceinline__ __device__ DType  __shfl_down_sync(unsigned, DType val, int delta) {
  return __shfl_down(val, delta);
}

// shuffle masks not used before CUDA 9.
#define CREATE_SHFL_MASK(mask, predicate) \
    unsigned mask = 0u;
#else
#define CREATE_SHFL_MASK(mask, predicate) \
    unsigned mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif





namespace cuda {
template<typename DType, int kFilterHeight, int kFilterWidth>
__global__ void __launch_bounds__(1024, 2)
DepthwiseConv2dForwardKernel(const DType* input,
                             const DType* filter,
                             const DepthwiseArgs args,
                             int num_outputs,
                             DType* output) {
  const int in_channel = args.in_channel;
  const int in_height = args.in_height;
  const int in_width = args.in_width;
  const int filter_height = kFilterHeight > 0 ? kFilterHeight : args.filter_height;
  const int filter_width = kFilterWidth > 0 ? kFilterWidth : args.filter_width;
  const int stride_height = args.stride_height;
  const int stride_width = args.stride_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;
  const int out_channel = args.out_channel;
  const int out_height = args.out_height;
  const int out_width = args.out_width;
  const int batch_filter_offset = in_channel * filter_height * filter_width;

  CUDA_KERNEL_LOOP(thread_id, num_outputs) {
    // Compute the indexes of this thread in the output.
    //
    // We want coalesced reads so we make sure that each warp reads
    // a contiguous chunk of memory.
    //
    // THIS IS PROBABLY WRONG, we are not doing coalesced reads
    // into the input, because of the depth multiplier division...
    const int out_w = thread_id % out_width;
    const int out_h = (thread_id / out_width) % out_height;
    const int out_c = (thread_id / out_width / out_height) % out_channel;
    const int out_b = thread_id / out_width / out_height / out_channel;
    const int in_c = out_c;
    

    // Data is stored in the following format (let's assume we
    // flatten the height and width into one contiguous dimension
    // called "P".
    //
    // B1C1P1 B1C1P2 ..... B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 ..... B2C2P1 B2C2P2 ....
    //
    // Each row contains in_channel * in_height * in_width values
    // for each sample in the batch.
    //
    // We can further flatten it into:
    //
    // B1C1P1 B1C1P2 .....
    // B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 .....
    // B2C2P1 B2C2P2 ....
    //
    // where each row is a contiguous array of all of the spatial
    // pixels for a given batch and input depth.  The following
    // loop unrolls across the filter dimensions for a given thread,
    // indexing into the filter value and the corresponding input
    // patch.
    //
    // We can compute the index into the patch once right here.
    const int input_offset_temp = (out_b * in_channel + in_c) * (in_height * in_width);
    const int filter_offset_temp = out_b * batch_filter_offset + in_c * filter_height * filter_width;

    // Finally, we can iterate over the spatial dimensions and perform the
    // convolution, writing into the output at the end.
    //
    // We perform an additional optimization, where we can determine
    // whether the patch fits within the image indices statically, and
    // avoid boundary checking within the loop.
    const int input_h_start = out_h * stride_height - pad_height;
    const int input_w_start = out_w * stride_width - pad_width;
    const int input_h_end = input_h_start + filter_height;
    const int input_w_end = input_w_start + filter_width;

    DType sum = 0;
    if (input_h_start >= 0 && input_w_start >= 0 &&
        input_h_end < in_height && input_w_end < in_width) {
      // Loop that doesn't need to check for boundary conditions.
      CUDA_UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
        const int in_h = input_h_start + f_h;
        const int filter_offset_h = filter_width * f_h;
        CUDA_UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
          const int in_w = input_w_start + f_w;
          const int input_offset = (input_offset_temp) + (in_h * in_width) + in_w;
          const int filter_offset = filter_offset_temp + filter_offset_h + f_w;
          sum += ldg(input + input_offset) * ldg(filter + filter_offset);
        }
      }
    } else {
      // Loop that needs to check for boundary conditions.
      CUDA_UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
        const int in_h = input_h_start + f_h;
        const int filter_offset_h = filter_width * f_h;
        CUDA_UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
          const int in_w = input_w_start + f_w;
          // TODO(vrv): the in_h check can be done outside of this loop;
          // benchmark both methods to determine the better decision.
          if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
            const int in_w = input_w_start + f_w;
            const int input_offset = input_offset_temp + (in_h * in_width) + in_w;
            const int filter_offset = filter_offset_temp + filter_offset_h + f_w;
            sum += ldg(input + input_offset) * ldg(filter + filter_offset);
          }
        }
      }
    }
    output[thread_id] = sum;
  }
}

template<typename DType>
__global__ void __launch_bounds__(640, 2)
DepthwiseConv2dBackwardDataKernel(const DepthwiseArgs args,
                                  const DType* out_grad,
                                  const DType* filter, DType* in_grad,
                                  int num_in_grad) {
  const int channel = args.in_channel;
  const int in_height = args.in_height;
  const int in_width = args.in_width;
  const int filter_height = args.filter_height;
  const int filter_width = args.filter_width;
  const int stride_height = args.stride_height;
  const int stride_width = args.stride_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;
  const int out_height = args.out_height;
  const int out_width = args.out_width;

  const int in_pixels = in_height * in_width;
  const int out_pixels = out_height * out_width;

  CUDA_KERNEL_LOOP(thread_id, num_in_grad) {
    // Compute the indexes of this thread in the input.
    const int in_w = thread_id % in_width;
    const int in_h = (thread_id / in_width) % in_height;
    const int channel_idx = (thread_id / in_width / in_height) % channel;
    const int batch_idx = thread_id / channel / in_width / in_height;
    DType sum = 0.0f;

    const int out_h_start = mxnet::common::cuda::CudaMax<int>(
        0, (in_h - filter_height + pad_height + stride_height) / stride_height);
    const int out_h_end = mxnet::common::cuda::CudaMin(
        out_height - 1, (in_h + pad_height) / stride_height);
    const int out_w_start = mxnet::common::cuda::CudaMax<int>(
            0, (in_w - filter_width + pad_width + stride_width) / stride_width);
    const int out_w_end = mxnet::common::cuda::CudaMin(
        out_width - 1, (in_w + pad_width) / stride_width);

    const int filter_offset_temp = channel_idx * filter_height * filter_width;
    const int out_grad_offset_temp = (batch_idx * channel * out_pixels) +
        (channel_idx * out_pixels);

    for (int out_h = out_h_start; out_h <= out_h_end; ++out_h) {
      const int f_h = in_h + pad_height - out_h * stride_height;
      const int filter_offset_h = filter_offset_temp + f_h * filter_width;
      const int out_grad_offset_h = out_grad_offset_temp + out_h * out_width;
      for (int out_w = out_w_start; out_w <= out_w_end; ++out_w) {
        const int f_w = in_w + pad_width - out_w * stride_width;
        const int filter_offset = filter_offset_h + f_w;
        const int out_grad_offset = out_grad_offset_h + out_w;
        sum += ldg(out_grad + out_grad_offset) * ldg(filter + filter_offset);
      }
    }
    const int in_grad_offset = (batch_idx * channel * in_pixels) +
        (channel_idx * in_pixels) + (in_h * in_width) + (in_w);
    in_grad[in_grad_offset] += sum;
  }
}

// A Cuda kernel to compute the depthwise convolution backprop w.r.t. filter.
template <typename DType, int kFilterWidth, int kFilterHeight>
__global__ void __launch_bounds__(640, 2)
DepthwiseConv2dBackwardFilterKernel(const DepthwiseArgs args,
                                    const DType* out_backprop,
                                    const DType* input,
                                    DType* filter_backprop,
                                    int num_out_backprop) {
  const int in_channel = args.in_channel;
  const int in_height = args.in_height;
  const int in_width = args.in_width;
  const int filter_height = kFilterHeight > 0 ? kFilterHeight : args.filter_height;
  const int filter_width = kFilterWidth > 0 ? kFilterWidth : args.filter_width;
  const int stride_height = args.stride_height;
  const int stride_width = args.stride_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;
  const int out_channel = args.out_channel;
  const int out_height = args.out_height;
  const int out_width = args.out_width;

  CUDA_KERNEL_LOOP(thread_id, num_out_backprop) {
    // Compute the indexes of this thread in the output.
    const int out_w = thread_id % out_width;
    const int out_h = (thread_id / out_width) % out_height;
    const int out_c = (thread_id / out_width / out_height) % out_channel;
    const int out_b = thread_id / out_width / out_height / out_channel;
    const int in_c = out_c;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int in_row_start = out_h * stride_height - pad_height;
    const int in_col_start = out_w * stride_width - pad_width;
    const int in_row_end = in_row_start + filter_height;
    const int in_col_end = in_col_start + filter_width;

    const int out_backprop_offset =
        (out_b * out_channel * out_height * out_width) +
        (out_c * out_height * out_width) + (out_h * out_width) +
        (out_w);

    const DType out_bp = ldg(out_backprop + out_backprop_offset);
    if (in_row_start >= 0 && in_col_start >= 0 &&
        in_row_end < in_height && in_col_end < in_width) {
      CUDA_UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
        const int in_row = in_row_start + f_h;
        // Avoid repeated computation.
        const int input_offset_temp =
            (out_b * in_channel * in_height * in_width) +
            (in_c * in_height * in_width) + (in_row * in_width);
        const int filter_backprop_temp =
            (in_c * filter_width * filter_height) +
            (filter_width * f_h);

        CUDA_UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
          const int in_col = in_col_start + f_w;
          const int input_offset = input_offset_temp + in_col;
          DType partial_sum = ldg(input + input_offset) * out_bp;
          DType* addr = filter_backprop + (filter_backprop_temp + f_w);
          atomicAdd(addr, partial_sum);
        }
      }
    } else {
      CUDA_UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
        const int in_row = in_row_start + f_h;
        // Avoid repeated computation.
        const int input_offset_temp =
            (out_b * in_channel * in_height * in_width) +
            (in_c * in_height * in_width) + (in_row * in_width);
        const int filter_backprop_temp =
            (in_c * filter_width * filter_height) +
            (filter_width * f_h);
        CUDA_UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
          const int in_col = in_col_start + f_w;

          if (in_row >= 0 && in_row < in_height && in_col >= 0 && in_col < in_width) {
            const int input_offset = input_offset_temp + in_col;
            DType partial_sum = ldg(input + input_offset) * out_bp;
            DType* addr = filter_backprop + (filter_backprop_temp + f_w);
            // Potentially many threads can add to the same address so we have
            // to use atomic add here.
            // TODO(jmchen): If atomic add turns out to be slow, we can:
            // 1. allocate multiple buffers for the gradients (one for each
            // example in a batch, for example). This can reduce the
            // contention on the destination; 2. Have each thread compute one
            // gradient for an element in the filters. This should work well
            // when the input depth is big and filter size is not too small.
            atomicAdd(addr, partial_sum);
          }
        }
      }
    }
  }
}


}  // namespace cuda




}  // namespace depthwise_conv
}  // namespace tf

#endif  // MXNET_OPERATOR_NN_DEPTHWISE_CONVOLUTION_TF_CUH_
