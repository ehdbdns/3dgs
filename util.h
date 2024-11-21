#pragma once
#define _CRT_SECURE_NO_WARNINGS
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include<Eigen/Dense>
#include<torch/torch.h>
#include<torch/script.h>
#include<iostream>
#include<cnpy.h>
#include<opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include<c10/cuda/CUDACachingAllocator.h>
#include <fstream>
#include<json.hpp>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include"3rdParty/simple_knn/spatial.h"
#include<torch/script.h>
#include<string>
#include <vector>
#include <stack>
#include <random>
#include <ctime>
#include <thread> // 包含线程库
#include <chrono> // 包含时间库
using slice = torch::indexing::Slice;
#define device torch::kCUDA
#define host torch::kCPU
#define PI 3.1415926f

struct cameraPose {
	torch::Tensor pose;//3x5 rotage+transform+hwf
	float nearBound;
	float farBound;
	cv::Mat cameraImg;
};



torch::Tensor inverse_sigmoid(torch::Tensor x);


torch::Tensor normalize(torch::Tensor x);


torch::Tensor build_rotation(torch::Tensor r);


torch::Tensor build_scaling_rotation(torch::Tensor s, torch::Tensor r);


torch::Tensor strip_lowerdiag(torch::Tensor L);


torch::Tensor build_covariance_from_scaling_rotation(torch::Tensor scaling, float scaling_modifier, torch::Tensor rotation);


void print(std::string desc, torch::Tensor t);


void print_size(torch::Tensor t);


torch::Tensor getWorld2View2(torch::Tensor R, torch::Tensor t);


torch::Tensor getWorld2ViewCamera(torch::Tensor R, torch::Tensor t);


torch::Tensor getWorld2ViewCamera2(torch::Tensor R, torch::Tensor t);


torch::Tensor getProjectionMatrix(float znear, float zfar, float fovX, float fovY);


double focal2fov(double focal, double pixels);


void resizeCamera(cameraPose* cp, float factor);


void resizeCameras(std::vector<cameraPose>& cameraposes, float factor);


torch::Tensor my_l1_loss(torch::Tensor network_output, torch::Tensor gt);


torch::Tensor gaussian(int window_size, float sigma);


torch::Tensor create_window(int window_size, int channel);


torch::Tensor _ssim(torch::Tensor img1, torch::Tensor img2, torch::Tensor window, int window_size, int channel);


torch::Tensor ssim(torch::Tensor img1, torch::Tensor img2);


torch::Tensor viewMatrix(torch::Tensor z, torch::Tensor up, torch::Tensor pos);


torch::Tensor TensorCross(torch::Tensor a, torch::Tensor b);


torch::Tensor qvec2rotmat( torch::Tensor& qvec);