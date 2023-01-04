#include <iostream>
#include <cstdlib>
#include <string>
#include "nanotrack.hpp"

using namespace std;
using namespace cv;

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

static float sz_whFun(cv::Point2f wh)
{
    float pad = (wh.x + wh.y) * 0.5f;
    float sz2 = (wh.x + pad) * (wh.y + pad);
    return std::sqrt(sz2);
}

static std::vector<float> sz_change_fun(std::vector<float> w, std::vector<float> h,float sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    std::vector<float> pad(rows * cols, 0);
    std::vector<float> sz2;
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            pad[i*cols+j] = (w[i * cols + j] + h[i * cols + j]) * 0.5f;
        }
    }
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            float t = std::sqrt((w[i * rows + j] + pad[i*rows+j]) * (h[i * rows + j] + pad[i*rows+j])) / sz;

            sz2.push_back(std::max(t,(float)1.0/t) );
        }
    }
    return sz2;
}

static std::vector<float> ratio_change_fun(std::vector<float> w, std::vector<float> h, cv::Point2f target_sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    float ratio = target_sz.x / target_sz.y;
    std::vector<float> sz2; 
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float t = ratio / (w[i * cols + j] / h[i * cols + j]);
            sz2.push_back(std::max(t, (float)1.0 / t));
        }
    }

    return sz2; 
}

NanoTrack::NanoTrack()
{   
    
    
}

NanoTrack::~NanoTrack()
{
    
}

void NanoTrack::init(cv::Mat img, cv::Rect bbox) 
{
    create_window(); 

    create_grids(); 

    cv::Point target_pos; // cx, cy
    cv::Point2f target_sz = {0.f, 0.f}; //w,h

    target_pos.x = bbox.x + bbox.width / 2;  
    target_pos.y = bbox.y + bbox.height / 2; 
    target_sz.x=bbox.width;
    target_sz.y=bbox.height;
    
    float wc_z = target_sz.x + cfg.context_amount * (target_sz.x + target_sz.y);
    float hc_z = target_sz.y + cfg.context_amount * (target_sz.x + target_sz.y);
    float s_z = round(sqrt(wc_z * hc_z));  

    cv::Scalar avg_chans = cv::mean(img);
    cv::Mat z_crop = get_subwindow_tracking(img, target_pos, cfg.exemplar_size, int(s_z), avg_chans); //cv::Mat BGR order

    this->net_backbone_interpreter->resizeTensor(net_backbone_input_tensor,
                                     1,
									 3,
									 cfg.exemplar_size,
									 cfg.exemplar_size);
    this->net_backbone_interpreter->resizeSession(net_backbone_session);

    std::shared_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(MNN::CV::BGR,
                                                                                 MNN::CV::RGB,
																				 mean_vals,
                                                                                 3,
																				 norm_vals,
                                                                                 3));
    process->convert(z_crop.data,
    				 cfg.exemplar_size,
					 cfg.exemplar_size,
					 z_crop.step[0],
					 net_backbone_input_tensor);


    net_backbone_interpreter->runSession(net_backbone_session);
    MNN::Tensor *zf = net_backbone_interpreter->getSessionOutput(net_backbone_session, "output");
    zf->printShape();

    zf_host = make_shared<MNN::Tensor>(new MNN::Tensor(zf, MNN::Tensor::CAFFE));
    zf->copyToHostTensor(zf_host.get());

    this->state.channel_ave=avg_chans;
    this->state.im_h=img.rows;
    this->state.im_w=img.cols;
    this->state.target_pos=target_pos;
    this->state.target_sz= target_sz;  
}

void NanoTrack::update(const cv::Mat &x_crops, cv::Point &target_pos, cv::Point2f &target_sz,  float scale_z, float &cls_score_max)
{
    this->net_backbone_interpreter->resizeTensor(net_backbone_input_tensor,
                                     1,
									 3,
									 cfg.instance_size,
									 cfg.instance_size);
    this->net_backbone_interpreter->resizeSession(net_backbone_session);

    std::shared_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(MNN::CV::BGR,
                                                                                 MNN::CV::RGB,
																				 mean_vals,
                                                                                 3,
																				 norm_vals,
                                                                                 3));
    process->convert(x_crops.data,
    				 cfg.instance_size,
					 cfg.instance_size,
					 x_crops.step[0],
					 net_backbone_input_tensor);

    net_backbone_interpreter->runSession(net_backbone_session);
    MNN::Tensor *xf = net_backbone_interpreter->getSessionOutput(net_backbone_session, "output");

    xf_host = make_shared<MNN::Tensor>(new MNN::Tensor(xf, MNN::Tensor::CAFFE));
    xf->copyToHostTensor(xf_host.get());

    net_head_input_tensor1->copyFromHostTensor(zf_host.get());
    net_head_input_tensor2->copyFromHostTensor(xf_host.get());

    net_head_interpreter->runSession(net_head_session);

    MNN::Tensor *cls_score = net_head_interpreter->getSessionOutput(net_head_session, "output1");
    MNN::Tensor *bbox_pred = net_head_interpreter->getSessionOutput(net_head_session, "output2");

    MNN::Tensor *cls_score_host = new MNN::Tensor(cls_score, MNN::Tensor::CAFFE);
    cls_score->copyToHostTensor(cls_score_host);

    MNN::Tensor *bbox_pred_host = new MNN::Tensor(bbox_pred, MNN::Tensor::CAFFE);
    bbox_pred->copyToHostTensor(bbox_pred_host);

    std::vector<float> cls_score_sigmoid; 
    cls_score_sigmoid.clear();
    
    int cols = cls_score->shape()[3];	//cls_score.w;
    int rows = cls_score->shape()[2];	//cls_score.h;
    float* cls_score_data = cls_score_host->host<float>() + cols*rows;

    for (int i = 0; i < cols*rows; i++)
    {        
        cls_score_sigmoid.push_back(sigmoid(cls_score_data[i]));
    }

    std::vector<float> pred_x1(cols*rows, 0), pred_y1(cols*rows, 0), pred_x2(cols*rows, 0), pred_y2(cols*rows, 0);

    float* bbox_pred_data1 = bbox_pred_host->host<float>();//bbox_pred.channel(0);
    float* bbox_pred_data2 = bbox_pred_host->host<float>() + 1* cols*rows;//bbox_pred.channel(1);
    float* bbox_pred_data3 = bbox_pred_host->host<float>() + 2* cols*rows;//bbox_pred.channel(2);
    float* bbox_pred_data4 = bbox_pred_host->host<float>() + 3* cols*rows;//bbox_pred.channel(3);
    
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {

            pred_x1[i*cols + j] = this->grid_to_search_x[i*cols + j] - bbox_pred_data1[i*cols + j];
            pred_y1[i*cols + j] = this->grid_to_search_y[i*cols + j] - bbox_pred_data2[i*cols + j];
            pred_x2[i*cols + j] = this->grid_to_search_x[i*cols + j] + bbox_pred_data3[i*cols + j];
            pred_y2[i*cols + j] = this->grid_to_search_y[i*cols + j] + bbox_pred_data4[i*cols + j];
        }
    }

    // size penalty  
    std::vector<float> w(cols*rows, 0), h(cols*rows, 0); 
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++) 
        {
            w[i*cols + j] = pred_x2[i*cols + j] - pred_x1[i*cols + j];
            h[i*rows + j] = pred_y2[i*rows + j] - pred_y1[i*cols + j];
        }
    }

    float sz_wh = sz_whFun(target_sz);
    std::vector<float> s_c = sz_change_fun(w, h, sz_wh);
    std::vector<float> r_c = ratio_change_fun(w, h, target_sz);

    std::vector<float> penalty(rows*cols,0);
    for (int i = 0; i < rows * cols; i++)
    {
        penalty[i] = std::exp(-1 * (s_c[i] * r_c[i]-1) * cfg.penalty_k);
    }

    // window penalty
    std::vector<float> pscore(rows*cols,0);
    int r_max = 0, c_max = 0; 
    float maxScore = 0; 

    for (int i = 0; i < rows * cols; i++)
    {
        pscore[i] = (penalty[i] * cls_score_sigmoid[i]) * (1 - cfg.window_influence) + this->window[i] * cfg.window_influence; 
        if (pscore[i] > maxScore) 
        {
            // get max 
            maxScore = pscore[i]; 
            r_max = std::floor(i / rows); 
            c_max = ((float)i / rows - r_max) * rows;  
        }
    }
    
    // to real size
    float pred_x1_real = pred_x1[r_max * cols + c_max]; // pred_x1[r_max, c_max]
    float pred_y1_real = pred_y1[r_max * cols + c_max];
    float pred_x2_real = pred_x2[r_max * cols + c_max];
    float pred_y2_real = pred_y2[r_max * cols + c_max];

    float pred_xs = (pred_x1_real + pred_x2_real) / 2;
    float pred_ys = (pred_y1_real + pred_y2_real) / 2;
    float pred_w = pred_x2_real - pred_x1_real;
    float pred_h = pred_y2_real - pred_y1_real;

    float diff_xs = pred_xs - cfg.instance_size / 2;
    float diff_ys = pred_ys - cfg.instance_size / 2;

    diff_xs /= scale_z; 
    diff_ys /= scale_z;
    pred_w /= scale_z;
    pred_h /= scale_z;

    target_sz.x = target_sz.x / scale_z;
    target_sz.y = target_sz.y / scale_z;

    // size learning rate
    float lr = penalty[r_max * cols + c_max] * cls_score_sigmoid[r_max * cols + c_max] * cfg.lr;

    // size rate
    auto res_xs = float (target_pos.x + diff_xs);
    auto res_ys = float (target_pos.y + diff_ys);
    float res_w = pred_w * lr + (1 - lr) * target_sz.x;
    float res_h = pred_h * lr + (1 - lr) * target_sz.y;

    target_pos.x = int(res_xs);
    target_pos.y = int(res_ys);

    target_sz.x = target_sz.x * (1 - lr) + lr * res_w;
    target_sz.y = target_sz.y * (1 - lr) + lr * res_h;

    cls_score_max = cls_score_sigmoid[r_max * cols + c_max];
}

void NanoTrack::track(cv::Mat im) 
{
    
    cv::Point target_pos = this->state.target_pos;
    cv::Point2f target_sz = this->state.target_sz;
    
    float hc_z = target_sz.y + cfg.context_amount * (target_sz.x + target_sz.y);
    float wc_z = target_sz.x + cfg.context_amount * (target_sz.x + target_sz.y);
    float s_z = sqrt(wc_z * hc_z);  
    float scale_z = cfg.exemplar_size / s_z;  

    float d_search = (cfg.instance_size - cfg.exemplar_size) / 2; 
    float pad = d_search / scale_z; 
    float s_x = s_z + 2*pad;

    cv::Mat x_crop;  
    x_crop  = get_subwindow_tracking(im, target_pos, cfg.instance_size, int(s_x),state.channel_ave);

    // update
    target_sz.x = target_sz.x * scale_z;
    target_sz.y = target_sz.y * scale_z;

    float cls_score_max;
    
    this->update(x_crop, target_pos, target_sz, scale_z, cls_score_max);

    target_pos.x = std::max(0, min(state.im_w, target_pos.x));
    target_pos.y = std::max(0, min(state.im_h, target_pos.y));
    target_sz.x = float(std::max(10, min(state.im_w, int(target_sz.x))));
    target_sz.y = float(std::max(10, min(state.im_h, int(target_sz.y))));

    state.target_pos = target_pos;
    state.target_sz = target_sz;
}


void NanoTrack::load_model(std::string model_backbone, std::string model_head)
{
    net_backbone_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_backbone.c_str()));
    MNN::ScheduleConfig config;
    config.numThread = 1;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
    config.backendConfig = &backendConfig;

    net_backbone_session = net_backbone_interpreter->createSession(config);
    net_backbone_input_tensor = net_backbone_interpreter->getSessionInput(net_backbone_session, nullptr);

    net_head_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_head.c_str()));
    net_head_session = net_head_interpreter->createSession(config);

    net_head_input_tensor1 = net_head_interpreter->getSessionInput(net_head_session, "input1");
    net_head_input_tensor2 = net_head_interpreter->getSessionInput(net_head_session, "input2");
}

// 生成每一个格点的坐标 
void NanoTrack::create_window()
{
    int score_size= cfg.score_size; 
    std::vector<float> hanning(score_size,0);  
    this->window.resize(score_size*score_size, 0);

    for (int i = 0; i < score_size; i++)
    {
        float w = 0.5f - 0.5f * std::cos(2 * 3.1415926535898f * i / (score_size - 1));
        hanning[i] = w;
    } 
    for (int i = 0; i < score_size; i++)
    {
        for (int j = 0; j < score_size; j++)
        {
            this->window[i*score_size+j] = hanning[i] * hanning[j]; 
        }
    }    
}

// 生成每一个格点的坐标 
void NanoTrack::create_grids()
{
    /*
    each element of feature map on input search image
    :return: H*W*2 (position for each element)
    */
    int sz = cfg.score_size;   //16x16

    this->grid_to_search_x.resize(sz * sz, 0);
    this->grid_to_search_y.resize(sz * sz, 0);

    for (int i = 0; i < sz; i++)
    {
        for (int j = 0; j < sz; j++)
        {
            this->grid_to_search_x[i*sz+j] = j*cfg.total_stride;   
            this->grid_to_search_y[i*sz+j] = i*cfg.total_stride;
        }
    }
}

cv::Mat NanoTrack::get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz,cv::Scalar channel_ave)
{
    float c = (float)(original_sz + 1) / 2;
    int context_xmin = std::round(pos.x - c);
    int context_xmax = context_xmin + original_sz - 1;
    int context_ymin = std::round(pos.y - c);
    int context_ymax = context_ymin + original_sz - 1;

    int left_pad = int(std::max(0, -context_xmin));
    int top_pad = int(std::max(0, -context_ymin));
    int right_pad = int(std::max(0, context_xmax - im.cols + 1));
    int bottom_pad = int(std::max(0, context_ymax - im.rows + 1));

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;
    cv::Mat im_path_original;

    if (top_pad > 0 || left_pad > 0 || right_pad > 0 || bottom_pad > 0)
    {
        cv::Mat te_im = cv::Mat::zeros(im.rows + top_pad + bottom_pad, im.cols + left_pad + right_pad, CV_8UC3);
       
        cv::copyMakeBorder(im, te_im, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, channel_ave);
        im_path_original = te_im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    else
        im_path_original = im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));

    cv::Mat im_path;
    cv::resize(im_path_original, im_path, cv::Size(model_sz, model_sz));

    return im_path; 
}
