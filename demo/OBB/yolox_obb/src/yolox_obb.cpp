#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <NvInferPlugin.h>
#include <NvInfer.h>
#include "obb_nms.cpp"
#include "logging.h"
#include "yolox_obb.h"
#include "math.h"

#define CHECK(status, msg, code) \
    do\
    {\
        auto ret_ = status;\
        if (!ret_)\
        {\
            std::cerr << msg << ": " << code << std::endl;\
            abort();\
        }\
    } while (0)

struct Object
{
    Object(const cv::RotatedRect & r, 
        const int & l, 
        const float & s): rect(r), label(l), score(s) {};
    cv::RotatedRect rect;
    int label;
    float score;
};

static Logger gLogger{Logger::Severity::kINFO};
using namespace nvinfer1;

#define mINFO Logger::Severity::kINFO 
#define mERROR Logger::Severity::kERROR
#define mWARNING Logger::Severity::kWARNING 

class NodeInfo
{
    public:
        NodeInfo(){name = "None"; index=-1; dataSize=0;};
        NodeInfo(const std::string & n): name(n), index(-1), dataSize(0) {}
        NodeInfo(const std::string & n, const long & i): name(n), index(i), dataSize(0) {}
        NodeInfo(const std::string & n, const Dims & d, const long & i): name(n), dim(d), index(i), dataSize(0){}
        std::string name;
        size_t getDataSize();
        Dims dim;
        long index;
    protected:
        size_t dataSize;
};

size_t NodeInfo::getDataSize()
{
    if (dataSize <= 0)
    {
        auto dataSize_ = 1;
        for (int i = 0; i < dim.nbDims; i++)
        {
            dataSize_ *= dim.d[i];
        }
        dataSize = dataSize_;
        return dataSize;
    }
    else
    {
        return dataSize;
    }
}

void loadEngine( 
    IRuntime * (&runtime), ICudaEngine * (&engine), 
    IExecutionContext * (&context), 
    NodeInfo & inputNode, std::vector<NodeInfo> & outputNodes,
    const std::string & ckpt) {

    char * trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file(ckpt, std::ios::binary);
    CHECK(file.good(), "File is not opened correctly", -1);
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    CHECK(trtModelStream != nullptr, "TRTModelStream data buffer is created failed", -1);
    file.read(trtModelStream, size);
    file.close();
    CHECK(initLibNvInferPlugins(&gLogger, ""), "Init NvInfer Plugin failed", -1);
    runtime = createInferRuntime(gLogger);
    CHECK(runtime != nullptr, "Runtime create failed", -1);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    CHECK(engine != nullptr, "Engine create failed", -1);
    context = engine->createExecutionContext();
    CHECK(context != nullptr, "Context create failed", -1);
    inputNode.index = engine->getBindingIndex(inputNode.name.c_str());
    inputNode.dim = engine->getBindingDimensions(inputNode.index);

    for (auto & node : outputNodes)
    {
        node.index = engine->getBindingIndex(node.name.c_str());
        node.dim = engine->getBindingDimensions(node.index);
    }
    delete [] trtModelStream;
}

void doInference(IExecutionContext * context,
                 const ICudaEngine * engine,
                 NodeInfo & inputNode, std::vector<NodeInfo> & outputNodes,
                 const cv::Mat & inputImage, float ** outputResults) {
    auto dataTypeSize = sizeof(float);
    const auto buffersNb = engine->getNbBindings();
    CHECK(buffersNb == (1 + outputNodes.size()), "Number of bind error", -1);
    void * buffers[buffersNb];
    float * input = new float[inputNode.getDataSize()];
    memcpy(input, inputImage.data, dataTypeSize * inputNode.getDataSize());
    int ret = cudaMalloc(&buffers[inputNode.index], inputNode.getDataSize() * dataTypeSize);
    CHECK(ret == cudaSuccess, "Malloc input buffer memory failed", ret);
    for (auto & node : outputNodes) {
        ret = cudaMalloc(&buffers[node.index], node.getDataSize() * dataTypeSize);
        CHECK(ret == cudaSuccess, "Malloc output buffer memory failed", ret);
    }
    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    CHECK(ret == cudaSuccess, "Create stream failed", ret);
    ret = cudaMemcpyAsync(buffers[inputNode.index], input, inputNode.getDataSize() * dataTypeSize, cudaMemcpyHostToDevice, stream);
    CHECK(ret == cudaSuccess, "Memcpy input failed", ret);
    ret = context->enqueue(1, buffers, stream, nullptr);
    CHECK(ret, "Inference in enqueue failed!", -1);
    for (auto i = 0; i < outputNodes.size(); i++) {
        ret = cudaMemcpyAsync(outputResults[i], buffers[outputNodes[i].index], 
            outputNodes[i].getDataSize() * dataTypeSize, cudaMemcpyDeviceToHost, stream);
        CHECK(ret == cudaSuccess, "Memcpy output failed", ret);
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    ret = cudaFree(buffers[inputNode.index]);
    CHECK(ret == cudaSuccess, "Input buffer free failed", ret);
    for (auto & node: outputNodes) {
        ret = cudaFree(buffers[node.index]);
        CHECK(ret == cudaSuccess, "Ouput buffer free failed", ret);
    }
    delete [] input; // NOTE:
}

cv::Mat dataPreprocess(cv::Mat & image, const std::vector<int> & resizeSize) {
    const int inputH = resizeSize[0];
    const int inputW = resizeSize[1];
    cv::Mat resizeImage(inputH, inputW, CV_8UC3);
    double r = std::min(inputH / (image.rows * 1.0), inputW / (image.cols * 1.0));
    int unpadW = r * image.cols;
    int unpadH = r * image.rows;
    cv::Mat re(unpadH, unpadW, CV_8UC3);
    cv::resize(image, re, re.size());
    re.copyTo(resizeImage(cv::Rect(0, 0, re.cols, re.rows)));
    cv::Mat blob;
    #ifdef OPENCV_DNN_CUDA
        blob = cv::dnn::blobFromImage(resizeImage);
    #else
        resizeImage.convertTo(resizeImage, CV_32F);
        blob.create({1, 3, inputH, inputW}, CV_32F);
        cv::Mat ch[3];
        for (int i = 0; i < 3; i++) {
            ch[i] = cv::Mat(inputH, inputW, CV_32F, blob.ptr(0, i));
        }
        cv::split(resizeImage, ch);
    #endif
    return blob;
}
           
void doPostprocess(
    std::vector<Object> & objects,
    std::vector<NodeInfo> & outputNodes,
    float ** outputResults,
    const float & conf_thre,
    const float & nms_thre,
    const int (&originSize)[2],
    const int (&resizeSize)[2]) {

    NodeInfo & boxNode = outputNodes[0];
    const float * boxesOut = outputResults[0];
    const float * scoresOut = outputResults[1];
    const float * classOut = outputResults[2];
    double scale_ratio = std::min(originSize[0] / resizeSize[0] * 1.0, 
        originSize[1] / resizeSize[1] * 1.0);
    const int & num = boxNode.dim.d[0];
    std::vector<int> order(num, 0);
    std::vector<int> keep(num, 0);
    std::vector<int> suppressed(num, 0);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), 
        [&scoresOut](int i1, int i2)->bool
        {return scoresOut[i1] > scoresOut[i2];});

    int64_t numKeep = 0;
	int i, j; float ovr, s;
    for (int _i = 0; _i < num; _i++) {
        i = order[_i];
        s = scoresOut[i];
        if (s < conf_thre) {
            suppressed[i] = 1; 
            continue;
        }
        if (suppressed[i] == 1) {
            continue;
        }
        keep[numKeep++] = i;
        for (auto _j = _i + 1; _j < num; _j++) {
            j = order[_j];
            s = scoresOut[j];
            if (s < conf_thre) {
                suppressed[j] = 1; 
                continue;
            }
            if (suppressed[j] == 1) {
                continue;
            }
            ovr = single_box_iou_rotated<float>(
                boxesOut + i * 5, boxesOut + j * 5
            );
            if (ovr >= nms_thre) {
                suppressed[j] = 1;
            }
        }
    }

    for (auto & i : keep) {
        auto ctrX = boxesOut[i * 5 + 0] / scale_ratio;
        auto ctrY = boxesOut[i * 5 + 1] / scale_ratio;
        auto width = boxesOut[i * 5 + 2] / scale_ratio;
        auto height = boxesOut[i * 5 + 3] / scale_ratio;
        auto theta = -boxesOut[i * 5 + 4] * 180 / M_PI; 
        auto score = scoresOut[i * 1 + 0];
        auto label = (int)classOut[i * 1 + 0];
        if (ctrX < 0. || ctrY < 0. || ctrX > originSize[1] || ctrY > originSize[0])
            continue;
        objects.emplace_back(
            (cv::RotatedRect) {
                (cv::Point2f) {ctrX, ctrY},
                (cv::Size2f) {width, height},
                theta
            },
            label, score
        );
    }
}

void doVisprocess(cv::Mat & image, 
                  std::vector<Object> & objects,
                  const std::string & savePath) {
    
    int numColors = sizeof(colorList) / sizeof(colorList[0]);
    for (auto & object : objects) {
        const float * colorPtr = colorList[object.label % numColors];
        cv::Scalar color = cv::Scalar(colorPtr[2], colorPtr[1], colorPtr[0]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txtColor = c_mean > 0.5 ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);
        char text[128]{0};
        snprintf(text, sizeof(text), "%s|%.3f", 
            dota10ClassNames[object.label], object.score);
        cv::Point2f ps[4];
        object.rect.points(ps);
        std::vector<std::vector<cv::Point>> tmpContours;
        std::vector<cv::Point> contours;
        for (auto i = 0; i != 4; i++) {
            contours.emplace_back(cv::Point2i(ps[i]));
        }
        tmpContours.insert(tmpContours.end(), contours);
        cv::drawContours(image, tmpContours, 0, color, 2);
        cv::putText(image, text, object.rect.center, cv::FONT_HERSHEY_SIMPLEX, 0.6, txtColor, 2, 1);
    }
    cv::imwrite(savePath, image);
}

void checkArgs(int & argc, char ** argv) {
    CHECK(argc == 5, "Args is None", -1);
}

int main(int argc, char ** argv) {
    checkArgs(argc, argv);
    const std::string ckptPath{argv[1]}; 
    const std::string imagePath{argv[2]};
    const std::string savePath{argv[3]};
    const int resizeSize = std::atoi((const char*)argv[4]);
    std::stringstream argsInfo;
    argsInfo << "\n[Args] ckpt: " << ckptPath << "\n"
             << "imagePath: " << imagePath << "\n"
             << "savePath: " << savePath << "\n"
             << "imageSize: " << resizeSize;
    gLogger.log(mINFO, argsInfo.str().c_str());
    IRuntime * runtime{nullptr}; ICudaEngine * engine{nullptr};
    IExecutionContext * context{nullptr}; 
    NodeInfo inputNode{"input"};
    std::vector<NodeInfo> outputNodes{{"boxes"}, {"scores"}, {"class"}};
    loadEngine(runtime, engine, context, inputNode, outputNodes, ckptPath);
    gLogger.log(mINFO, "Load engine success");
    cv::Mat image = cv::imread(imagePath);
    int originSize[2] = {image.rows, image.cols};
    cv::Mat blobImage = dataPreprocess(image, {resizeSize, resizeSize});
    float * outputResults[outputNodes.size()];
    for (auto i = 0; i < outputNodes.size(); i++) {
        outputResults[i] = new float[outputNodes[i].getDataSize()];
    }
    auto startTime = std::chrono::system_clock::now();
    doInference(context, engine, inputNode, outputNodes, blobImage, outputResults);
    gLogger.log(mINFO, "Do inference successs");
    std::vector<Object> objects;
    auto inferTime = std::chrono::system_clock::now();
    doPostprocess(objects, outputNodes, outputResults, 
        0.1, 0.1, originSize, {resizeSize, resizeSize});
    gLogger.log(mINFO, "Do doPostprocess successs");
    auto postTime = std::chrono::system_clock::now();
    doVisprocess(image, objects, savePath);
    auto endTime = std::chrono::system_clock::now();
    char finishInfo[256];
    snprintf(finishInfo, sizeof(finishInfo), "Inference time: %d ms, Postprocess time: %d ms, Total time: %d ms", 
        std::chrono::duration_cast<std::chrono::milliseconds>(inferTime - startTime).count(),
        std::chrono::duration_cast<std::chrono::milliseconds>(postTime - inferTime).count(),
        std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count());
    gLogger.log(mINFO, finishInfo);
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

