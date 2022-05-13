#include "yolox_obb.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <dirent.h>
#include <exception>
#include <opencv2/opencv.hpp>
// #include <xtensor/xarray.hpp>

const char * input_blob_name = "input_0";
const char * output_blob_name = "output_0";
static const int num_classes = 18;
static const int input_size[2] = {1024, 1024}; 
static const int strides[3] = {8, 16, 32};
static Logger gLogger;

using namespace nvinfer1;

cv::Mat static_resize(cv::Mat & img, const int resize_size[2]);
void decode_outputs(float * output, std::vector<Object> & objects, 
                    const float scale, const int * origin_size, 
                    const int * input_size, const int * strides, 
                    const int & num_classes, const int num_stages=3,
                    const float bbox_thre=0.3, const float nms_thre=0.45);

void draw_objects(cv::Mat & image, 
                  const std::vector<Object> & objects, 
                  const std::string save_path = "",
                  const bool is_save=true);

float* blobFromImage(cv::Mat & img);

void PutRotatedText(cv::Mat & img,
                    char * text,
                    const cv::RotatedRect & rotatedrect,
                    const cv::Scalar & color, int thickness,
                    int lineType, int baseLine);

void DrawRotatedRect(cv::Mat & img,
                     const cv::RotatedRect & rotatedrect,
                     const cv::Scalar & color, int thickness, 
                     int lineType);

void doInference(IExecutionContext & context, 
                 float * input, float * output, 
                 const int input_data_size,
                 const int output_data_size,
                 const int inputIndex,
                 const int outputIndex);

int main(int argc, char** argv)
{
    char * trtModelStream(nullptr);    
    size_t size{0};
    if (argc == 6 && std::string(argv[2]) == "-i" && std::string(argv[4]) == "-s")
    {
        const std::string engine_file_path {argv[1]};
        std::cout << "engine file is " << engine_file_path << std::endl;

        std::ifstream file(engine_file_path, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    }
    else
    {
        std::cerr << "some errors in argv" << std::endl;
        std::cerr << "please check argv" << std::endl;
        abort();
    }
    // argv info
    const std::string input_path {argv[3]};
    const std::string save_path {argv[5]};
    std::cout << "input path is " << input_path << std::endl;
    std::cout << "save path is " << save_path << std::endl;
    std::string suffixStr = input_path.substr(input_path.find_last_of('.') + 1);

    // create engine and conetext
    IRuntime * runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine * engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext * context = engine->createExecutionContext();
    assert(context != nullptr);

    delete[] trtModelStream;

    // get size of input and output
    const int inputIndex = engine->getBindingIndex(input_blob_name);
    auto input_data_size = 3 * input_size[0] * input_size[1];

    const int outputIndex = engine->getBindingIndex(output_blob_name);
    auto out_dims = engine->getBindingDimensions(outputIndex);
    auto output_data_size = 1;
    for(int j=0; j<out_dims.nbDims; ++j)
        output_data_size *= out_dims.d[j];

    // image inference
    if (suffixStr == "jpg" || suffixStr == "png")
    {
        float * prob = new float[output_data_size];
        std::cout << "Begin inference images ..." << std::endl;
        cv::Mat img = cv::imread(input_path);
        const int origin_size[2] = {img.rows, img.cols};
        float scale_ratio = std::min(input_size[0] / (origin_size[0] * 1.0), 
            input_size[1] / (origin_size[1] * 1.0));

        cv::Mat pr_img = static_resize(img, input_size);
        float * blob = blobFromImage(pr_img);

        auto start = std::chrono::system_clock::now();
        doInference(*context, blob, prob, input_data_size, output_data_size, inputIndex, outputIndex);

        auto end = std::chrono::system_clock::now();
        std::cout << "Inference cost :" << \
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << \
            "ms" << std::endl;
        std::vector<Object> objects;
        start = std::chrono::system_clock::now();
        decode_outputs(prob, 
                       objects, 
                       scale_ratio, 
                       origin_size, 
                       input_size, 
                       strides, 
                       num_classes, 
                       NUM_STAGES, BBOX_THRE, NMS_THRE);
        end = std::chrono::system_clock::now();
        std::cout << "Decode cost :" << \
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << \
            "ms" << std::endl;
        draw_objects(img, objects, save_path, true);
        delete [] blob;
        delete [] prob;
    }
    else if (suffixStr == "mp4" || suffixStr == "avi")
    {
        std::cout << "Begin inference videos ..." << std::endl;
        cv::VideoCapture capture;
        capture.open(input_path);
        if (!capture.isOpened())
        {
            std::cout << "Can not open ..." << std::endl;
            return -1;
        }    
        cv::Size size = cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT));
        cv::VideoWriter writer;
        // writer.open(save_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, size, true);
        writer.open(save_path, 0x7634706d, 10, size, true);
        cv::Mat frame;
        const int origin_size[2] = {size.height, size.width};
        float scale_ratio = std::min(input_size[0] / origin_size[0], input_size[1] / origin_size[1]);
        while (capture.read(frame))
        {
            auto start = std::chrono::system_clock::now();
            float * prob = new float[output_data_size];
            cv::Mat pr_img = static_resize(frame, input_size); 
            float * blob = blobFromImage(pr_img);
            doInference(*context, blob, prob, input_data_size, output_data_size, inputIndex, outputIndex);
            std::vector<Object> objects;
            decode_outputs(prob, 
                           objects, 
                           scale_ratio, 
                           origin_size, 
                           input_size, 
                           strides, 
                           num_classes, 
                           NUM_STAGES, BBOX_THRE, NMS_THRE);
            draw_objects(frame, objects, "", false);
            auto end = std::chrono::system_clock::now();
            int inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            char FPS_text[256];
            sprintf(FPS_text, "FPS: %.1f", inference_time * 0.025);
            cv::Size label_size = cv::getTextSize(FPS_text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, 0);
            cv::putText(frame, FPS_text, (cv::Point2f){10.f, 10.f + (float)label_size.height}, 
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
            writer.write(frame);
            cv::waitKey(10);
            delete [] prob;
            delete [] blob;
        }
    }
    context->destroy();
    engine->destroy(); 
    runtime->destroy();
    return 0;
}

void doInference(IExecutionContext & context, 
                 float * input, float * output, 
                 const int input_data_size,
                 const int output_data_size,
                 const int inputIndex,
                 const int outputIndex)
{
    const ICudaEngine & engine = context.getEngine();
    assert(engine.getNbBindings() == 2);
    void * buffers[2];
    // const int mBatchSize = engine.getMaxBatchSize();
    CHECK(cudaMalloc(&buffers[inputIndex], input_data_size * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_data_size * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, input_data_size * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_data_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int get_num_anchors(const int * strides, const int * size, const int num_stages)
{
    std::vector<int> feature_size;
    int h = size[0];
    int w = size[1];
    for (int i = 0; i < num_stages; i++)
        feature_size.push_back(h * w / (strides[i] * strides[i]));
    return std::accumulate(feature_size.begin(), feature_size.end(), 0);
}


static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        Object object = objects[i];
        float object_width = object.rect.size.width;
        float object_height = object.rect.size.height;
        areas[i] = object_width * object_height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object & a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            // intersection over union
            std::vector<cv::Point2f> rotatedinterregion;
            try {
                cv::rotatedRectangleIntersection(a.rect, b.rect, rotatedinterregion);
            }
            catch(const char * &e) {
                std::cout << "there are existing boxes overrided" << std::endl;
                keep = 1 ;
                continue;
            }
            std::vector<cv::Point2f> hull;
            if (rotatedinterregion.size() < 3)
                continue;
            cv::convexHull(cv::Mat(rotatedinterregion), hull, true);
            if (hull.size() < 3)
                continue;
            float inter_area = cv::contourArea(hull);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void decode_outputs(float * output, std::vector<Object> & objects, 
                    const float scale, const int * origin_size, 
                    const int * input_size, const int * strides, 
                    const int & num_classes, const int num_stages,
                    const float bbox_thre, const float nms_thre)
{
    // std::vector<Object> proposals;
    const size_t num_anchors = get_num_anchors(strides, input_size, num_stages);
    const size_t out_channels = num_classes + 6;
    // generate grids
    // auto grides = xt::empty<float>(grids_shape);
    float * grides = new float[num_anchors * 3];
    int index = 0;
    for (int i = 0; i < num_stages; i++)
    {
        int num_grid_y = input_size[0] / strides[i];
        int num_grid_x = input_size[1] / strides[i];
        for (int g1 = 0; g1 < num_grid_y; g1++)
        {
            for (int g0 = 0; g0 < num_grid_x; g0++)
            {
                grides[index * 3 + 0] = (float)g0;
                grides[index * 3 + 1] = (float)g1;
                grides[index * 3 + 2] = (float)strides[i];
                index++;
            }
        }
    }
    // decodea prob
    std::vector<Object> proposals;
    for (int i = 0; i < num_anchors; i++)
    {
        Object object;
        float ctr_x = *(output + out_channels * i + 0);
        float ctr_y = *(output + out_channels * i + 1);
        float w = *(output + out_channels * i + 2);
        float h = *(output + out_channels * i + 3);
        float angle = *(output + out_channels * i + 4);
        float obj_prob = *(output + out_channels * i + 5);
        int class_pred = std::distance(
            output + out_channels * (i + 1) - num_classes, 
            std::max_element(output + out_channels * (i + 1) - num_classes, output + out_channels * (i + 1)));
        float class_prob = *(output + out_channels * (i + 1) - num_classes + class_pred);
        float prob = obj_prob * class_prob;
        if (prob < bbox_thre)
            continue;
        float grid_x = grides[i * 3 + 0];
        float grid_y = grides[i * 3 + 1];
        float stride = grides[i * 3 + 2];
        ctr_x = (ctr_x + grid_x) * stride;
        ctr_y = (ctr_y + grid_y) * stride;
        w = std::exp(w) * stride;
        h = std::exp(h) * stride;
        angle = -angle * 180 / M_PI;
        object.rect =  (cv::RotatedRect){(cv::Point2f){ctr_x, ctr_y}, (cv::Size2f){w, h}, angle};
        object.label = class_pred;
        object.prob = prob;
        proposals.push_back(object);
    }
    std::vector<int> picked;
    qsort_descent_inplace(proposals);
    nms_sorted_bboxes(proposals, picked, nms_thre);
    // objects.resize(picked.size());
    for (int i = 0; i < picked.size(); i++)
    {
        Object object = proposals[picked[i]];
        float ctr_x = object.rect.center.x / scale;
        float ctr_y = object.rect.center.y / scale;
        float width = object.rect.size.width / scale;
        float height = object.rect.size.height / scale;
        if (ctr_x < 0. || ctr_y < 0. || ctr_x > origin_size[1] || ctr_y > origin_size[0]) 
            continue;
        objects.push_back(
            (Object){
                (cv::RotatedRect){
                    (cv::Point2f){ctr_x, ctr_y},
                    (cv::Size2f){width, height},
                    object.rect.angle
                },
                object.label,
                object.prob
            }
        );
    }
}

cv::Mat static_resize(cv::Mat & img, const int resize_size[2])
{
    const int input_h = resize_size[0];
    const int input_w = resize_size[1];
    float r = std::min(input_w / (img.cols * 1.0), input_h / (img.rows * 1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3); // 8UC3(uint8 and 3 channels)
    cv::resize(img, re, re.size());
    cv::Mat out(input_h, input_w, CV_8UC3);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

float* blobFromImage(cv::Mat& img){
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                // cv::Vec3b的b指的是数据类型(b是8U)
                blob[c * img_w * img_h + h * img_w + w] =
                    (float)img.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
    return blob;
}

void draw_objects(
    cv::Mat & image, 
    const std::vector<Object> & objects, 
    const std::string save_path,
    const bool is_save)
{
    for (size_t i = 0; i < objects.size(); i++)    
    {
        const Object & obj = objects[i];

        cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5)
            txt_color = cv::Scalar(0, 0, 0);
        else
            txt_color = cv::Scalar(255, 255, 255);
        char text[256];
        DrawRotatedRect(image, obj.rect, color, 4, 8);
        sprintf(text, "%s %.1f%", dota20_class_names[obj.label], obj.prob);
        // PutRotatedText(image, text, obj.rect, txt_color, 1, 1);
        cv::putText(image, text, obj.rect.center, cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1, 1);
        if (is_save)
            cv::imwrite(save_path, image);
        // cv::imshow("det_res.jpg", image);
        // cv::waitKey(0);
    }
}

void DrawRotatedRect(cv::Mat & img,
                     const cv::RotatedRect & rotatedrect,
                     const cv::Scalar & color, int thickness, 
                     int lineType)
{
  // 提取旋转矩形的四个角点
	cv::Point2f ps[4];
	rotatedrect.points(ps); 
  // 构建轮廓线
	std::vector<std::vector<cv::Point>> tmpContours;    // 创建一个InputArrayOfArrays 类型的点集
	std::vector<cv::Point> contours;
	for (int i = 0; i != 4; ++i) {
		contours.emplace_back(cv::Point2i(ps[i]));
	}
	tmpContours.insert(tmpContours.end(), contours); 
  // 绘制轮廓，即旋转矩形
	cv::drawContours(img, tmpContours, 0, color,thickness, lineType);
}

void PutRotatedText(cv::Mat & img,
                    char * text,
                    const cv::RotatedRect & rotatedrect,
                    const cv::Scalar & color, int thickness,
                    int lineType, int baseLine)
{
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1., &baseLine);
    cv::Scalar txt_bk_color = color * 0.7 * 255;
    cv::Point2f center = rotatedrect.center;
    // cv::Size2f wh = rotatedrect.size;
    cv::Mat img_t = cv::Mat::zeros(img.rows, img.cols, img.type());
    // cv::rectangle(img_t, (cv::Rect){(cv::Point2f)(center.x - wh.width / 2.0, center.y - wh.height / 2.0 - label_size.height-1), 
    //     (cv::Size2f(label_size.width, label_size.height + baseLine))}, txt_bk_color, 
    //     thickness, lineType);
    cv::putText(img_t, text, cv::Point(center.x, center.y), cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
    // cv::Mat M = cv::getRotationMatrix2D(ps[0], rotatedrect.angle, 1.0);    
    // int _len = std::max(img_t.cols, img_t.rows);
    // cv::warpAffine(img_t, img_t, M, cv::Size(_len, _len));
    cv::addWeighted(img, 1.0, img_t, 0.6, 0., img);
}
