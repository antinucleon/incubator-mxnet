/*!
 * Copyright (c) 2015 by Contributors
 * \file spatial_upsampling.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_NN_BLOCK_UPSAMPLE_EXPR_H_
#define MXNET_OPERATOR_NN_BLOCK_UPSAMPLE_EXPR_H_

#include <mshadow/extension.h>

namespace mshadow {
namespace expr {

/*! \brief nearest neighboor upsampling
 *         out(x, y) = in(int(x / scale_x), int(y / scale_y))
 *  \tparam SrcExp source expression
 *  \tparam DType data type
 *  \tparam srcdim source dimension
 */
template<typename SrcExp, typename DType, int srcdim>
struct UpSamplingBlockExp :
  public MakeTensorExp<UpSamplingBlockExp<SrcExp, DType, srcdim>,
                       SrcExp, srcdim, DType> {
  /*! \brief source oprand */
  const SrcExp &src_;
  /*! \brief up sampling scale */
  index_t scale_h_;
  index_t scale_w_;
  /*! \brief constructor */
  UpSamplingBlockExp(const SrcExp &src, index_t scale_h, index_t scale_w)
    : src_(src), scale_h_(scale_h), scale_w_(scale_w) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    this->shape_[srcdim - 2] *= scale_h_;
    this->shape_[srcdim - 1] *= scale_w_;
  }
};


template<typename SrcExp, typename DType, int etype>
inline UpSamplingBlockExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
upsampling_block(const Exp<SrcExp, DType, etype> &src, index_t scale_h, index_t scale_w) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
    ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return UpSamplingBlockExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), scale_h, scale_w);
}

template<typename SrcExp, typename DType, int srcdim>
struct Plan<UpSamplingBlockExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const UpSamplingBlockExp<SrcExp, DType, srcdim> &e)
    : src_(MakePlan(e.src_)),
      scale_h_(e.scale_h_),
      scale_w_(e.scale_w_),
      new_height_(e.shape_[srcdim - 2]),
      src_height_(static_cast<index_t>(e.shape_[srcdim - 2] / e.scale_h_)) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t x = j;
    const index_t y = i % new_height_;
    const index_t c = i / new_height_;
    const index_t h = static_cast<index_t>(y / scale_h_);
    const index_t w = static_cast<index_t>(x / scale_w_);
    return src_.Eval(c * src_height_ + h, w);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t scale_h_;
  const index_t scale_w_;
  const index_t new_height_;
  const index_t src_height_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MXNET_OPERATOR_NN_BLOCK_UPSAMPLE_EXPR_H_
