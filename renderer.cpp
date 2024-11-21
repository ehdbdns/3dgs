#pragma once
#include "renderer.h"
#include"ParameterConfig.h"
#include"scene.h"
#include"util.h"
torch::autograd::tensor_list render(Camera camera, GaussianModel& gs, torch::Tensor bg, float scaling_modifier) {


	torch::Tensor screenspace_points = torch::zeros_like(gs.get_xyz().to(torch::kFloat32)).to(device);
	screenspace_points.set_requires_grad(true);

	float tanfovx = tan(camera.FoVx * 0.5);
	float tanfovy = tan(camera.FoVy * 0.5);


	auto means3D = gs.get_xyz();
	auto means2D = screenspace_points;
	auto opacity = gs.get_opacity();
	auto scales = gs.get_scaling();
	auto rotations = gs.get_rotation();
	auto shs = gs.get_features();
	torch::Tensor colors_precomp = torch::tensor({},torch::kFloat32).to(device);
	torch::Tensor cov3D_precomp = torch::tensor({},torch::kFloat32).to(device);
	auto retList= rasterize_gaussians( means3D,
		means2D,
	 shs,
	colors_precomp,
		opacity,
	 scales,
		rotations,
	cov3D_precomp,
		camera.image_height,
		camera.image_width,
		 tanfovx,
		 tanfovy,
		 bg,
		 scaling_modifier,
		camera.world_view_transform,
		camera.full_proj_transform,
		gs.active_sh_degree,
		camera.camera_center,
		false,
	    true);//debug模式可以输出更多错误信息
	retList.push_back(means2D);
	return retList;
}