#ifndef MXNET_OPERATOR_NN_CONTRIB_BATCH_DEPTHWISE_H_
#define MXNET_OPERATOR_NN_CONTRIB_BATCH_DEPTHWISE_H_



struct DepthwiseArgs {
  // Input layer dimensions
  int batch;
  int in_height;
  int in_width;
  int in_channel;
  int filter_height;
  int filter_width;
  int stride_height;
  int stride_width;
  int pad_height;
  int pad_width;

  // Output layer dimensions
  int out_height;
  int out_width;
  int out_channel;
};


#endif // MXNET_OPERATOR_NN_CONTRIB_BATCH_DEPTHWISE_H_