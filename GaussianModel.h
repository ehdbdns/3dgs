#pragma once
#include"util.h"
#include"ParameterConfig.h"
class GaussianModel:torch::nn::Module {
public:
    GaussianModel() = default;
	GaussianModel(int SHdegree);
	void setup_functions();
    torch::Tensor get_scaling();
    torch::Tensor get_rotation();
    torch::Tensor& get_xyz();
    torch::Tensor get_features();
    torch::Tensor get_opacity();
    torch::Tensor get_covariance(float scaling_modifier);
    void createFromPcd(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,float spatial_lr_scale);
    void training_setup(OptimizationParameters&op);
    void add_densification_stats(torch::Tensor viewspace_point_tensor, torch::Tensor update_filter);
    void densify_and_prune(float max_grad, float min_opacity, float extent, int max_screen_size);
    void densify_and_clone(torch::Tensor grads, float grad_threshold, float scene_extent);
    void densify_and_split(torch::Tensor grads, float grad_threshold, float scene_extent);
    std::vector<torch::Tensor> GaussianModel::cat_tensors_to_optimizer(std::vector<torch::Tensor>& tensors_vec,torch::optim::Adam* optimizer);
    void densification_postfix(torch::Tensor new_xyz, torch::Tensor new_features_dc, torch::Tensor new_features_rest, torch::Tensor new_opacities, torch::Tensor new_scaling, torch::Tensor new_rotation);
    std::vector<torch::Tensor> _prune_optimizer(torch::Tensor mask_index);
    void prune_points(torch::Tensor mask);
    void reset_opacity();
    torch::Tensor replace_opacity_to_optimizer(torch::Tensor tensor);
    void saveCkpt(std::string base, std::string expname, int global_step, int epoch);
    void loadCkpt(std::string base, std::string expname, int* global_step, int* epoch);
    std::string find_max_tar_file(const std::string& basedir, const std::string& expname, const std::string& objName);
    void expend_gs_and_optimizer(int loadGsNum, int step);
    float get_lr_from_iteration(int step, float lr_init, float lr_final);
    void update_lr_xyz(int step);

    int active_sh_degree;
    int max_sh_degree;
    float percent_dense;
    double spatial_lr_scale=1.0;
    float position_lr_init;
    float position_lr_final;
    torch::Tensor        xyz;
    torch::Tensor        features_dc;
    torch::Tensor        features_rest;
    torch::Tensor        scaling;
    torch::Tensor        rotation;
    torch::Tensor        opacity;
    torch::Tensor        max_radii2D;
    torch::Tensor        xyz_gradient_accum;
    torch::Tensor        denom;
    std::unique_ptr<torch::optim::Adam>   optimizer;
    std::function<torch::Tensor(const torch::Tensor&)> scaling_activation;
    std::function<torch::Tensor(const torch::Tensor&)> scaling_inverse_activation;
    std::function<torch::Tensor(const torch::Tensor&, float, const torch::Tensor&)> covariance_activation;
    std::function<torch::Tensor(const torch::Tensor&)> opacity_activation;
    std::function<torch::Tensor(const torch::Tensor&)> inverse_opacity_activation;
    std::function<torch::Tensor(const torch::Tensor&)> rotation_activation;
    std::vector<void*>paramTensorBaseImpl;
};


