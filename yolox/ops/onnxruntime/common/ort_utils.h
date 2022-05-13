#ifndef _ORT_UTILS_H
#define _ORT_UTILS_H
#include <onnxruntime_cxx_api.h>

struct OrtTensorDimensions: std::vector<int64_t> {
    OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue * value)
    {
        OrtTensorTypeAndShapeInfo * info = ort.GetTensorTypeAndShape(value);
        std::vector<int64_t>::operator=(ort.GetTensorShape(info));
        ort.ReleaseTensorTypeAndShapeInfo(info);
    }
};
#endif