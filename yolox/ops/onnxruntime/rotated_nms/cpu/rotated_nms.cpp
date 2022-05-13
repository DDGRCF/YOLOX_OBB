#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>  // std::iota
#include <vector>
#include "ort_utils.h"
#include "rotated_nms.h"
#include "rotated_utils.hpp"


RotatedNmsKernel::RotatedNmsKernel(OrtApi api, const OrtKernelInfo *info)
    : api_(api), ort_(api_), info_(info) {
    iou_threshold_ = ort_.KernelInfoGetAttribute<float>(info, "iou_threshold");
    score_threshold_ = ort_.KernelInfoGetAttribute<float>(info, "score_threshold");
    small_threshold_ = ort_.KernelInfoGetAttribute<float>(info, "small_threshold");
    max_num_ = ort_.KernelInfoGetAttribute<int64_t>(info, "max_num");
    // create allocator
    allocator_ = Ort::AllocatorWithDefaultOptions();
}


void RotatedNmsKernel::Compute(OrtKernelContext *context) {
    const int64_t max_num = max_num_;
    const float small_threshold = small_threshold_;
    const float score_threshold = score_threshold_;
    const float iou_threshold = iou_threshold_;

    const OrtValue *boxes = ort_.KernelContext_GetInput(context, 0);
    const float *boxes_data =
        reinterpret_cast<const float *>(ort_.GetTensorData<float>(boxes));
    const OrtValue *scores = ort_.KernelContext_GetInput(context, 1);
    const float *scores_data =
        reinterpret_cast<const float *>(ort_.GetTensorData<float>(scores));

    OrtTensorDimensions boxes_dim(ort_, boxes);
    OrtTensorDimensions scores_dim(ort_, scores);

    int64_t nboxes = boxes_dim[0];
    assert(boxes_dim[1] == 5);

    // allocate tmp memory
    float *tmp_boxes = (float *)allocator_.Alloc(sizeof(float) * nboxes * 5);
    float *sc = (float *)allocator_.Alloc(sizeof(float) * nboxes);
    bool *select = (bool *)allocator_.Alloc(sizeof(bool) * nboxes);
    for (int64_t i = 0; i < nboxes; i++) {
        select[i] = true;
    }
    memcpy(tmp_boxes, boxes_data, sizeof(float) * nboxes * 5);
    memcpy(sc, scores_data, sizeof(float) * nboxes);

    // sort scores
    std::vector<float> tmp_sc;
    for (int i = 0; i < nboxes; i++) {
        tmp_sc.push_back(sc[i]);
    }
    std::vector<int64_t> order(tmp_sc.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&tmp_sc](int64_t id1, int64_t id2) {
        return tmp_sc[id1] > tmp_sc[id2];
    });


    for (int64_t _i = 0; _i < nboxes; _i++) {
        if (select[_i] == false) continue;
        auto i = order[_i];
        if (sc[i] < score_threshold){
            select[_i] = false;
            continue;
        } 
        auto tmp_box1 = tmp_boxes + i * 5;

        for (int64_t _j = _i + 1; _j < nboxes; _j++) {
            if (select[_j] == false) continue;
            auto j = order[_j];
            if (sc[j] < score_threshold){
                select[_j] = false;
                continue;
            }
            auto tmp_box2 = tmp_boxes + j * 5;
            auto ovr = single_box_iou_rotated(tmp_box1, tmp_box2, small_threshold);
            if (ovr > iou_threshold) select[_j] = false;
        }
    }
    std::vector<int64_t> res_order;
    for (int i = 0; i < nboxes; i++) {
        if (select[i]) {
        res_order.push_back(order[i]);
        }
    }

    auto len_res_data = std::min(static_cast<int64_t>(res_order.size()), max_num);
    std::vector<int64_t> inds_dims({len_res_data});

    OrtValue *res = ort_.KernelContext_GetOutput(context, 0, inds_dims.data(),
                                                inds_dims.size());
    int64_t *res_data = ort_.GetTensorMutableData<int64_t>(res);

    memcpy(res_data, res_order.data(), sizeof(int64_t) * inds_dims.size());
}