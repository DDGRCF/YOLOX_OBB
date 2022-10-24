#include <layer.h>
#include <net.h>

#include <stdio.h>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "obb_nms.hpp"

struct Object
{
    Object() = default;
    Object(const cv::RotatedRect & r,
           const int & l,
           const float & s): rect(r), label(l), prob(s) {}

    cv::RotatedRect rect;
    int label;
    float prob;
};

struct RunnerParameter {
    int target_size;
    float conf_thre;
    float nms_thre;
    bool agnostic;
    std::vector<std::string> class_names = {
        "large-vehicle", "swimming-pool", "helicopter", "bridge",
        "plane", "ship", "soccer-ball-field", "basketball-court",
        "ground-track-field", "small-vehicle", "baseball-diamond",
        "tennis-court", "roundabout", "storage-tank", "harbor"
    };
};

class YOLOXOBBDect {

public:
    YOLOXOBBDect() {
        state = 0;
    };
    YOLOXOBBDect(const std::string & param_path, const std::string & bin_path, const RunnerParameter & rp){
        yolox.opt.use_vulkan_compute = true;
        if ((state = yolox.load_param(param_path.c_str())) != 0) {
            return;
        }
        if ((state = yolox.load_model(bin_path.c_str())) != 0) {
            return;
        }
        runner_params = rp;
    }
    ~YOLOXOBBDect() = default;

public:
    int load_structure(const std::string & param_path, const std::string & bin_path) {
        yolox.clear(); 
        if (yolox.load_param(param_path.c_str()) != 0) {
            return -1;
        }

        if (yolox.load_model(bin_path.c_str()) != 0) {
            return -1;
        }
    }

    void load_run_params(const RunnerParameter & rp) {
        runner_params = rp;
    }

    int get_state() {
        return state;
    }

public:
    int detect(const std::string & image_path, std::vector<Object> & objects);
    int inference(const cv::Mat & image, std::vector<Object> & objects);
    int visualize(const cv::Mat & image, const std::vector<Object> & objects);
private:
    int state;
    ncnn::Net yolox;
    RunnerParameter runner_params;

private:
    static void qsort_descent_inplace(std::vector<Object> & objects);
    static void qsort_descent_inplace(std::vector<Object> & objects, int left, int right);
    static void nms_sorted_bboxes(const std::vector<Object> & faceobjects, 
                                  std::vector<int> & picked, const float nms_threshold, 
                                  const bool agnostic = false);
};

int YOLOXOBBDect::detect(const std::string & image_path, std::vector<Object> & objects) {
    int ret;
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        return -1;
    }
    ret = inference(image, objects);
    if (ret != 0) {
        return -1;
    }
    ret = visualize(image, objects);
    if (ret != 0) {
        return -1;
    }
    return 0;
}

int YOLOXOBBDect::inference(const cv::Mat & image, std::vector<Object> & objects) {
    int image_w = image.cols; 
    int image_h = image.rows;

    int w = image_w;
    int h = image_h;
    float scale = 1.f;

    if (w > h) {
        scale = (float) runner_params.target_size / w;
        w = runner_params.target_size;
        h = h * scale;
    } else {
        scale = (float) runner_params.target_size / h;
        h = runner_params.target_size;
        w = w  * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, image_w, image_h, w, h);

    int w_pad = (w + 31) / 32 * 32 - w; 
    int h_pad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, 0, h_pad, 0, w_pad, ncnn::BORDER_CONSTANT, 114.f);

    ncnn::Extractor ex = yolox.create_extractor();

    ncnn::Mat scores_out;
    ncnn::Mat class_out;
    ncnn::Mat boxes_out;

    ex.input("input", in_pad);
    ex.extract("scores", scores_out);
    ex.extract("boxes", boxes_out);
    ex.extract("class", class_out);

    auto num_proposals = scores_out.total();
    auto num_classes = class_out.w;
    std::vector<Object> proposals;
    for (int y = 0; y < (long)num_proposals; y++) {
        float* scores_row = scores_out.row(y);
        float* boxes_row = boxes_out.row(y);
        float* class_row = class_out.row(y);
        auto score = *scores_row;
        int class_index = std::max_element(class_row, class_row + num_classes) - class_row;
        auto class_score = class_row[class_index];
        score *= (float)(score * class_score);
        if (score < runner_params.conf_thre) continue;
        auto ctr_x = boxes_row[0] / scale; auto ctr_y = boxes_row[1] / scale;
        if (ctr_x < 0. || ctr_y < 0. || 
            ctr_x > runner_params.target_size || 
            ctr_y > runner_params.target_size) continue;
        
        auto obj_w = boxes_row[2] / scale; auto obj_h = boxes_row[3] / scale; auto obj_t = boxes_row[4];
        obj_t = -obj_t * 180 / M_PI;

        proposals.emplace_back(
            (cv::RotatedRect) {
                (cv::Point2f) {ctr_x, ctr_y},
                (cv::Size2f) {obj_w, obj_h},
                obj_t
            },
            class_index, score
        );
    }

    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, runner_params.nms_thre, runner_params.agnostic);

    auto keep_num = picked.size();
    objects.resize(keep_num);
    for (int i = 0; i < (long)keep_num; i++) {
        objects[i] = proposals[picked[i]];
    }
    return 0;
}

int YOLOXOBBDect::visualize(const cv::Mat & image, const std::vector<Object> & objects) {
    float color_array[3] = {190, 150, 37}; 
    cv::Scalar box_color = cv::Scalar(color_array[0], color_array[1], color_array[2]);
    cv::Scalar txt_color = cv::Scalar(0, 0, 0);
    for (auto & object : objects) {
        char text[128]{0};
        snprintf(text, sizeof(text), "%s|%.3f", 
            runner_params.class_names[object.label].c_str(), object.prob);
        cv::Point2f ps[4];
        object.rect.points(ps);
        std::vector<std::vector<cv::Point>> tmp_contours;
        std::vector<cv::Point> contours;
        for (auto i = 0; i != 4; i++) {
            contours.emplace_back(cv::Point2i(ps[i]));
        }
        tmp_contours.insert(tmp_contours.end(), contours);
        cv::putText(image, text, object.rect.center, cv::FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2, 1);
        cv::drawContours(image, tmp_contours, 0, box_color, 2);
    }
    cv::imwrite("yolox_demo.jpg", image);
    return 0;
}

void YOLOXOBBDect::qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void YOLOXOBBDect::qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

void YOLOXOBBDect::nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_thre, bool agnostic)
{
    picked.clear();

    const int n = faceobjects.size();

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float box1[5]; float box2[5];
            box1[0] = a.rect.center.x; box1[1] = a.rect.center.y;
            box1[2] = a.rect.size.width; box1[3] = a.rect.size.height;
            box1[4] = -a.rect.angle * M_PI / 180.;

            box2[0] = b.rect.center.x; box2[1] = b.rect.center.y;
            box2[2] = b.rect.size.width; box2[3] = b.rect.size.height;
            box2[4] = -b.rect.angle * M_PI / 180.;

            float iou = single_box_iou_rotated<float>(box1, box2, 0);
            if (iou > nms_thre)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s [param_path] [bin_path] [image_path]\n", argv[0]);
        return -1;
    }

    const char* param_path = argv[1];
    const char* bin_path = argv[2];
    const char* image_path = argv[3];

    RunnerParameter runner_params = {
        1024,           // target image size
        0.2,            // conf thre
        0.1,            // nms thre
        false,          // agnostic
    };

    YOLOXOBBDect yolox_dect(param_path, bin_path, runner_params);
    if (yolox_dect.get_state() != 0) {
        fprintf(stderr, "initilize model fail!");
        return 0;
    }
    std::vector<Object> objects;
    if (yolox_dect.detect(image_path, objects) != 0) {
        fprintf(stderr, "detect image fail!");
        return -1;
    }
    return 0;
}