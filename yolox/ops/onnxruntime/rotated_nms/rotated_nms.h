#ifndef _ROTATED_NMS_H
#define _ROTATED_NMS_H
#include <onnxruntime_cxx_api.h>


struct RotatedNmsKernel {
    RotatedNmsKernel(OrtApi api, const OrtKernelInfo *info);

    void Compute(OrtKernelContext *context);

    protected:
    OrtApi api_;
    Ort::CustomOpApi ort_;
    const OrtKernelInfo *info_;
    Ort::AllocatorWithDefaultOptions allocator_;

    float iou_threshold_;
    float score_threshold_;
    float small_threshold_;
    int64_t max_num_;
};

struct RotatedNmsOp : Ort::CustomOpBase<RotatedNmsOp, RotatedNmsKernel> {
  void *CreateKernel(OrtApi api, const OrtKernelInfo *info) const {
    return new RotatedNmsKernel(api, info);
  };

  const char *GetName() const { return "RotatedNonMaxSuppression"; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }

  // force cpu
  const char *GetExecutionProviderType() const {
    return "CPUExecutionProvider";
  }
};


#endif