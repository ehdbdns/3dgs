#pragma once
#include "diff_gaussian_rasterization.h"
#include"rasterize_points.h"
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
	bool debug){
	auto retVarList = RasterizeGaussiansFunc::apply(means3D,//apply函数ff返回的类型是std::vector<torch::Tensor>,即使forward函数只返回一个torch::Tensor也是如此
		means2D,
		sh,
		colors_precomp,
		opacities,
		scales,
		rotations,
		cov3Ds_precomp,
		bg,
		world_view_transform,
		full_proj_transform,
		camera_center,
		image_height,
		image_width,
		tanfovx,
		tanfovy,
		scaling_modifier,
		active_sh_degree,
		false,
		debug);
	return retVarList;
}



torch::autograd::tensor_list RasterizeGaussiansFunc::forward(
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
) {
	
	std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ret = RasterizeGaussiansCUDA(bg,
		means3D,
		colors_precomp,
		opacities,
		scales,
		rotations,
		scaling_modifier,
		cov3Ds_precomp,
		world_view_transform,
		full_proj_transform,
		tanfovx,
		tanfovy,
		image_height,
		image_width,
		sh,
		active_sh_degree,
		camera_center,
		false,
		debug);

	ctx->save_for_backward({ colors_precomp,//这个函数只能存张量
		means3D,
		scales,
		rotations,
		cov3Ds_precomp,
		std::get<2>(ret),//raddi
		sh,
		std::get<3>(ret),//geomBuf
		std::get<4>(ret),//binbuf
		std::get<5>(ret),//imgbuf
		bg,
		world_view_transform,
		full_proj_transform,
		camera_center,
		});
	ctx->saved_data["num_rendered"] = std::get<0>(ret);
	ctx->saved_data["scaling_modifier"] = scaling_modifier;
	ctx->saved_data["tanfovx"] = tanfovx;
	ctx->saved_data["tanfovy"] = tanfovy;
	ctx->saved_data["active_sh_degree"] = active_sh_degree;
	torch::Tensor color = std::get<1>(ret);
	torch::Tensor radii = std::get<2>(ret);
	return {color,radii};
}



torch::autograd::tensor_list  RasterizeGaussiansFunc::backward(
	torch::autograd::AutogradContext* ctx,
	torch::autograd::tensor_list grad_outputs) {

	// 从ctx中恢复保存的张量和设置
	auto saved = ctx->get_saved_variables();
	auto colors_precomp = saved[0];
	auto means3D = saved[1];
	auto scales = saved[2];
	auto rotations = saved[3];
	auto cov3Ds_precomp = saved[4];
	auto radii = saved[5];
	auto sh = saved[6];
	auto geomBuffer = saved[7];
	auto binningBuffer = saved[8];
	auto imgBuffer = saved[9];


	auto num_rendered= ctx->saved_data["num_rendered"].to<int>();
	auto scaling_modifier = ctx->saved_data["scaling_modifier"].to<float>();
	auto tanfovx = ctx->saved_data["tanfovx"].to<float>();
	auto tanfovy = ctx->saved_data["tanfovy"].to<float>();
	auto active_sh_degree = ctx->saved_data["active_sh_degree"].to<int>();

	// 调用CUDA函数来计算梯度
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>ret =
		RasterizeGaussiansBackwardCUDA(//调用backward后来到这个方法计算梯度
			saved[10],//bg
			means3D,
			radii,
			colors_precomp,
			scales,
			rotations,
			scaling_modifier,
			cov3Ds_precomp,
			saved[11],//world_view_transform
			saved[12],//full_proj_transform
			tanfovx,
			tanfovy,
			grad_outputs[0],
			sh,
			active_sh_degree,
			saved[13],//camera_center
			geomBuffer,
			num_rendered,
			binningBuffer,
			imgBuffer,
			true);


	torch::Tensor grad_means2D = std::get<0>(ret);//gmeans2D
	torch::Tensor grad_colors_precomp = std::get<1>(ret);//gcolors
	torch::Tensor grad_opacities = std::get<2>(ret);//gopacity
	torch::Tensor grad_means3D = std::get<3>(ret);//gmeans3D
	torch::Tensor grad_cov3Ds_precomp = std::get<4>(ret);//gcov3D
	torch::Tensor grad_sh = std::get<5>(ret);//gsh
	torch::Tensor grad_scales = std::get<6>(ret);//gscales
	torch::Tensor grad_rotations = std::get<7>(ret);//grotation

	torch::autograd::tensor_list tensorList= { grad_means3D, grad_means2D, grad_sh, grad_colors_precomp, grad_opacities, grad_scales, grad_rotations, grad_cov3Ds_precomp };//与forward输入一一对应
	for (int i = 0;i < 4;i++)
		tensorList.push_back(torch::tensor({0}));
	for (int i = 0;i < 8;i++)//对于非Tensor的输入，必须返回torch::Tensor()
		tensorList.push_back(torch::Tensor());
		return tensorList;
}