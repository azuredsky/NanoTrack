#ifndef NANOTRACK_H
#define NANOTRACK_H

#include <vector> 
#include <map>  
 
#include <opencv2/opencv.hpp>

#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "MNN/ImageProcess.hpp"

#define PI 3.1415926 

struct Config{ 
    
    std::string windowing = "cosine";
    std::vector<float> window;

    int stride = 16;
    float penalty_k = 0.148;
    float window_influence = 0.462;
    float lr = 0.390;
    int exemplar_size=127;
    int instance_size=255;
    int total_stride=16;
    int score_size=16;
    float context_amount = 0.5;
};

struct State { 
    int im_h; 
    int im_w;  
    cv::Scalar channel_ave; 
    cv::Point target_pos; 
    cv::Point2f target_sz = {0.f, 0.f}; 
    float cls_score_max; 
};

class NanoTrack {

public: 
    
    NanoTrack();
    
    ~NanoTrack(); 

    void init(cv::Mat img, cv::Rect bbox);
    
    void update(const cv::Mat &x_crops, cv::Point &target_pos, cv::Point2f &target_sz, float scale_z, float &cls_score_max);
    
    void track(cv::Mat im);
    
    void load_model(std::string model_backbone, std::string model_head);

    std::shared_ptr<MNN::Interpreter> net_backbone_interpreter;
    std::shared_ptr<MNN::Interpreter> net_head_interpreter;

    MNN::Session *net_backbone_session = nullptr;
    MNN::Tensor *net_backbone_input_tensor = nullptr;

    MNN::Session *net_head_session = nullptr;
    MNN::Tensor *net_head_input_tensor1 = nullptr;
    MNN::Tensor *net_head_input_tensor2 = nullptr;

    std::shared_ptr<MNN::Tensor> xf_host;
    std::shared_ptr<MNN::Tensor> zf_host;

    int stride=16;
    
    // state  dynamic
    State state;
    
    // config static
    Config cfg; 

    const float mean_vals[3] = { 0.f, 0.f, 0.f };
    const float norm_vals[3] = {1.f, 1.f, 1.f};

private: 
    void create_grids(); 
    void create_window();  
    cv::Mat get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz,cv::Scalar channel_ave);

    std::vector<float> grid_to_search_x;
    std::vector<float> grid_to_search_y;
    std::vector<float> window;
};

#endif 
