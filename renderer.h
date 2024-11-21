#pragma once
#include "util.h"
#include"diff_gaussian_rasterization.h"
#include"scene.h"
#include"GaussianModel.h"

torch::autograd::tensor_list render(Camera camera,GaussianModel& gs,torch::Tensor bg,float scaling_modifier);