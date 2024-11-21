#pragma once
#include"util.h"



class OptimizationParameters {
public:
	OptimizationParameters()
		: iterations(30000),
		position_lr_init(0.00016),
		position_lr_final(0.0000016),
		position_lr_delay_mult(0.01),
		position_lr_max_steps(30000),
		feature_lr(0.0025),
		opacity_lr(0.05),
		scaling_lr(0.005),
		rotation_lr(0.001),
		percent_dense(0.01),
		lambda_dssim(0.2),
		densification_interval(100),
		opacity_reset_interval(3000),
		densify_from_iter(500),
		densify_until_iter(15000),
		densify_grad_threshold(0.0002),
		random_background(false) {}

	int iterations;
	double position_lr_init;
	double position_lr_final;
	double position_lr_delay_mult;
	int position_lr_max_steps;
	double feature_lr;
	double opacity_lr;
	double scaling_lr;
	double rotation_lr;
	double percent_dense;
	double lambda_dssim;
	int densification_interval;
	int opacity_reset_interval;
	int densify_from_iter;
	int densify_until_iter;
	double densify_grad_threshold;
	bool random_background;
};


class ModelParameters {
public:
	int sh_degree;
	std::string _source_path;
	std::string _model_path;
	std::string _images;
	int _resolution;
	bool _white_background;
	std::string data_device;
	bool eval;
	ModelParameters() : sh_degree(3),
		_source_path(""),
		_model_path("C:\\mytinyrenderproj\\MyRenderer\\3dgs\\points3D.ply"),
		_images("images"),
		_resolution(-1),
		_white_background(false),
		data_device("cuda"),
		eval(false) {}
};


class GaussianRasterizationSettings {
public:
	int image_height;
	int image_width;
	float tanfovx;
	float tanfovy;
	torch::Tensor bg;
	float scale_modifier;
	torch::Tensor viewmatrix;
	torch::Tensor projmatrix;
	int sh_degree;
	torch::Tensor campos;
	bool prefiltered;
	bool debug;
	GaussianRasterizationSettings() = default;

	GaussianRasterizationSettings::GaussianRasterizationSettings(
		int image_height,
		int image_width,
		float tanfovx,
		float tanfovy,
		torch::Tensor bg,
		float scale_modifier,
		torch::Tensor viewmatrix,
		torch::Tensor projmatrix,
		int sh_degree,
		torch::Tensor campos,
		bool prefiltered,
		bool debug
	) :
		image_height(image_height),
		image_width(image_width),
		tanfovx(tanfovx),
		tanfovy(tanfovy),
		bg(bg),
		scale_modifier(scale_modifier),
		viewmatrix(viewmatrix),
		projmatrix(projmatrix),
		sh_degree(sh_degree),
		campos(campos),
		prefiltered(prefiltered),
		debug(debug)
	{};
};