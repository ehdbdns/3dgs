#pragma once
#include"util.h"
#include"ParameterConfig.h"


torch::autograd::tensor_list rasterize_gaussians(torch::Tensor& means3D,
	torch::Tensor means2D,
	torch::Tensor sh,
	torch::Tensor colors_precomp,
	torch::Tensor opacities,
	torch::Tensor scales,
	torch::Tensor rotations,
	torch::Tensor cov3Ds_precomp,
	int image_height,
	int image_width,
	float tanfovx,
	float tanfovy,
	torch::Tensor bg,
	const float scaling_modifier,
	torch::Tensor world_view_transform,
	torch::Tensor full_proj_transform,
	int active_sh_degree,
	torch::Tensor camera_center,
	bool prefiltered,
	bool debug);


class RasterizeGaussiansFunc :public torch::autograd::Function<RasterizeGaussiansFunc> {
public:

	static torch::autograd::tensor_list RasterizeGaussiansFunc::forward(
		torch::autograd::AutogradContext* ctx,
		torch::Tensor means3D,
		torch::Tensor means2D,
		torch::Tensor sh,
		torch::Tensor colors_precomp,
		torch::Tensor opacities,
		torch::Tensor scales,
		torch::Tensor rotations,
		torch::Tensor cov3Ds_precomp,
		torch::Tensor bg,
		torch::Tensor world_view_transform,
		torch::Tensor full_proj_transform,
		torch::Tensor camera_center,
		int image_height,
		int image_width,
		float tanfovx,
		float tanfovy,
		const float scaling_modifier,
		int active_sh_degree,
		bool prefiltered,
		bool debug
	);

	
	static torch::autograd::tensor_list backward(
		torch::autograd::AutogradContext* ctx,
		torch::autograd::tensor_list grad_outputs
	);


};