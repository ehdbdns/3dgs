#pragma once
#include"GaussianModel.h"
#include"util.h"
GaussianModel::GaussianModel(int SHdegree) {
    active_sh_degree = 0;
    max_sh_degree = SHdegree;
    xyz = torch::empty(0);
    features_dc = torch::empty(0);
    features_rest = torch::empty(0);
    scaling = torch::empty(0);
    rotation = torch::empty(0);
    opacity = torch::empty(0);
    max_radii2D = torch::empty(0);
    xyz_gradient_accum = torch::empty(0);
    denom = torch::empty(0);
    percent_dense = 0;
    spatial_lr_scale = 0;
    setup_functions();
}


void GaussianModel::setup_functions() {
    scaling_activation = torch::exp;
    scaling_inverse_activation = torch::log;
    covariance_activation = build_covariance_from_scaling_rotation;
    opacity_activation = torch::sigmoid;
    inverse_opacity_activation = inverse_sigmoid;
    rotation_activation = normalize;
}


torch::Tensor GaussianModel::get_scaling() {
    return scaling_activation(scaling);
}


torch::Tensor  GaussianModel::get_rotation() {
    return rotation_activation(rotation);
}


torch::Tensor&  GaussianModel::get_xyz() {
    return xyz;
}


torch::Tensor  GaussianModel::get_features() {
    features_dc = features_dc;
    features_rest = features_rest;
    if (features_dc.sizes()[0] == 0)
        return torch::Tensor();
    return torch::cat({ features_dc, features_rest }, 1);
}


torch::Tensor  GaussianModel::get_opacity() {
    return opacity_activation(opacity);
}


torch::Tensor  GaussianModel::get_covariance(float scaling_modifier) {
    return covariance_activation(get_scaling(), scaling_modifier, rotation);
}


void GaussianModel::createFromPcd(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float spatial_lr_scale) {


	Eigen::MatrixXf points_matrix = Eigen::MatrixXf::Zero(cloud->points.size(), 6);


	for (size_t i = 0; i < cloud->points.size(); ++i) {
		points_matrix(i, 0) = cloud->points[i].x;
		points_matrix(i, 1) = cloud->points[i].y;
		points_matrix(i, 2) = cloud->points[i].z;
		points_matrix(i, 3) = cloud->points[i].r;
		points_matrix(i, 4) = cloud->points[i].g;
		points_matrix(i, 5) = cloud->points[i].b;
	}



	float* data_ptr = points_matrix.data();
	std::vector<float> data_vec(data_ptr, data_ptr + points_matrix.size());
	torch::Tensor points_tensor = torch::from_blob((void*)data_vec.data(), { (int64_t)cloud->points.size(), 6 });
	points_tensor = points_tensor.toType(torch::kFloat32).to(device);
    auto  tensors = torch::split(points_tensor, 3, 1);
    torch::Tensor  posTensor = tensors[0];
    torch::Tensor colorTensor = tensors[1];


    max_sh_degree = 3;
    torch::Tensor  features = torch::zeros({ colorTensor.sizes()[0],3, static_cast<int64_t>( std::pow(max_sh_degree + 1., 2.)) }, torch::kFloat32).to(device);
    features.index({ slice(),slice(0,3),slice(0,1) }) = colorTensor.unsqueeze(2);//features的形状为N，3，sh   这里将每个3xsh的矩阵的第一列赋值为颜色
    features.index({slice(),slice(3,torch::indexing::None)}) = 0.0;//这里其实啥都没干
    torch::Tensor dist2_tensor =torch::clamp( distCUDA2(posTensor),0.0000001);
    torch::Tensor scales = torch::zeros_like(dist2_tensor).unsqueeze(1).repeat({ 1, 3 }).to(device) - 4;//torch::log(torch::sqrt(dist2_tensor)).unsqueeze(1).repeat({ 1, 3 }).to(device);//
    torch::Tensor  rots = torch::zeros({ posTensor.sizes()[0], 4 }).to(device);
    rots.index({ slice(),slice(0,1) }) = torch::ones({posTensor.sizes()[0],1});
    torch::Tensor  opacities = inverse_sigmoid(0.1 * torch::ones({ posTensor.sizes()[0], 1 }, torch::kFloat32)).to(device);
    xyz= register_parameter("_xyz", posTensor,true);//由于优化器的param必须是已经注册的张量，所以这里必须注册
    scaling= register_parameter("_scaling", scales,true);
    rotation= register_parameter("_ratations", rots,true);
    opacity= register_parameter("_opacity", opacities,true);
    features_dc = register_parameter("_features_dc", features.index({ slice(),slice(),slice(0,1) }).permute({0,2,1}), true);
    features_rest= register_parameter("_features_rest", features.index({ slice(),slice(),slice(1,features.sizes()[2])}).permute({0,2,1}), true);
    max_radii2D = torch::zeros({ get_xyz().sizes()[0] }).to(device);
}


void GaussianModel::training_setup(OptimizationParameters& op) {


    percent_dense = op.percent_dense;
    xyz_gradient_accum = torch::zeros({ get_xyz().sizes()[0], 1}).to(device);
    denom = torch::zeros({ get_xyz().sizes()[0], 1}).to(device);


	std::vector<torch::optim::OptimizerParamGroup> param_groups;
	param_groups.emplace_back(torch::optim::OptimizerParamGroup({ xyz }, std::make_unique<torch::optim::AdamOptions>(op.position_lr_init * spatial_lr_scale)));
	param_groups.emplace_back(torch::optim::OptimizerParamGroup({ features_dc }, std::make_unique<torch::optim::AdamOptions>(op.feature_lr)));
	param_groups.emplace_back(torch::optim::OptimizerParamGroup({ features_rest }, std::make_unique<torch::optim::AdamOptions>(op.feature_lr / 20.0)));
	param_groups.emplace_back(torch::optim::OptimizerParamGroup({ opacity }, std::make_unique<torch::optim::AdamOptions>(op.opacity_lr)));
	param_groups.emplace_back(torch::optim::OptimizerParamGroup({ scaling }, std::make_unique<torch::optim::AdamOptions>(op.scaling_lr)));
	param_groups.emplace_back(torch::optim::OptimizerParamGroup({ rotation }, std::make_unique<torch::optim::AdamOptions>(op.rotation_lr)));



     optimizer = std::make_unique<torch::optim::Adam>(param_groups, torch::optim::AdamOptions(0));

     position_lr_init = op.position_lr_init;
     position_lr_final = op.position_lr_final;

}


void GaussianModel::add_densification_stats(torch::Tensor viewspace_point_tensor, torch::Tensor update_filter) {
    xyz_gradient_accum.index_put_({ update_filter },xyz_gradient_accum.index_select({ 0 }, update_filter) + torch::norm(viewspace_point_tensor.grad().index_select({ 0 }, update_filter).index({ slice(),slice(0,2) }), 2, -1, true));
    denom.index_put_({ update_filter }, denom.index_select({ 0 }, update_filter) + 1);
}


void GaussianModel::densify_and_prune(float max_grad, float min_opacity, float extent, int max_screen_size) {
	auto grads = xyz_gradient_accum / denom;
    grads = torch::where(torch::isnan(grads), torch::zeros_like(grads), grads);
    densify_and_clone(grads, max_grad, extent);
    densify_and_split(grads, max_grad, extent);
    //剪枝
    auto selected_pts_mask = get_opacity().le(min_opacity).to(torch::kByte).squeeze();
    if (max_screen_size){
		torch::Tensor big_points_vs = max_radii2D > max_screen_size;
        std::cout << "由于raddi剪枝了" << big_points_vs.nonzero().sizes()[0] << "个高斯" << std::endl;
		torch::Tensor big_points_ws = std::get<0>(get_scaling().max(1))>  0.1*extent;
        std::cout << "由于scale剪枝了" << big_points_ws.nonzero().sizes()[0] << "个高斯" << std::endl;
        selected_pts_mask = torch::logical_or(torch::logical_or(selected_pts_mask, big_points_vs),big_points_ws);
    }
    int num = xyz.sizes()[0];
    prune_points(selected_pts_mask);
    std::cout << "剪枝了" << xyz.sizes()[0] - num << "个高斯" << std::endl;
    c10::cuda::CUDACachingAllocator::emptyCache();
}


void GaussianModel::densify_and_clone(torch::Tensor grads, float grad_threshold, float scene_extent) {
    auto selected_pts_mask = torch::norm(grads, 2,-1).ge(grad_threshold).to(torch::kByte);
    torch::Tensor condition = std::get<0>(torch::max(get_scaling(), 1)) <= (percent_dense * scene_extent);
    selected_pts_mask = torch::logical_and(selected_pts_mask, condition).to(torch::kInt32);
	auto index = torch::nonzero(torch::where(selected_pts_mask > 0, torch::ones_like(selected_pts_mask), torch::zeros_like(selected_pts_mask))).squeeze();
    auto new_xyz = xyz.index_select({ 0 }, index).clone();
    auto new_features_dc = features_dc.index_select({ 0 }, index).clone();
    auto new_features_rest = features_rest.index_select({ 0 }, index).clone();
    auto new_opacities = opacity.index_select({ 0 }, index).clone();
    auto new_scaling = scaling.index_select({ 0 }, index).clone();
    auto new_rotation = rotation.index_select({ 0 }, index).clone();
    densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation);
}


std::vector<torch::Tensor> GaussianModel::cat_tensors_to_optimizer(
    std::vector<torch::Tensor>& tensors_vec,
	torch::optim::Adam* optimizer) {

    std::vector<torch::Tensor> optimizable_tensors;
    int i = 0;
	for (auto& group : optimizer->param_groups()){
		assert(group.params().size() == 1);
        auto extension_tensor = tensors_vec[i];
        auto param = group.params()[0];


        auto& state = optimizer->state();//切记需要在step后才能得到每个param对应的OptimizerParamState，比如这个优化器里有6个参数，那么这个state表的size就为6，key为参数的unsafeGetTensorImpl，value为OptimizerParamState的uniqueptr，OptimizerParamState可以找到exp_avg等
        auto& param_state = state.at(param.unsafeGetTensorImpl());

        auto d = dynamic_cast<torch::optim::AdamParamState*>(param_state.get());//param_state是一个基类，不能直接得到exp_avg，所以得dynamic_cast转换到子类

        torch::Tensor exp_avg=d->exp_avg();
        torch::Tensor exp_avg_sq=d->exp_avg_sq();
        auto zeros_like_extension = torch::zeros_like(extension_tensor);
        exp_avg = torch::cat({ exp_avg, zeros_like_extension }, 0);
        exp_avg_sq = torch::cat({ exp_avg_sq, zeros_like_extension }, 0);
        d->exp_avg(exp_avg);
        d->exp_avg_sq(exp_avg_sq);
        auto stored_state = *d;

        param = torch::cat({ param, extension_tensor }, 0);
        param.set_requires_grad(true);
        optimizable_tensors.push_back(param);
		auto num = state.erase(group.params()[0].unsafeGetTensorImpl());
        state.insert({(void*) param.unsafeGetTensorImpl(),std::make_unique<torch::optim::AdamParamState>(stored_state) });

        group.params()[0] = param;//这里的赋值操作会导致    group.params()[0] 的TensorBase改变，而当step时如果paramgroup中出现了一个新的TensorBase会自动创建一个新的paramState 加进去，所以在赋值后应该删除原始paramState，再将stored_state加进去，这样step之后就不会自动增加了
        i++;
	}
	return optimizable_tensors;
}

void GaussianModel::expend_gs_and_optimizer(int loadGsNum,int step) {//由于load优化器只会读到state，而不会读到参数，所以这里初始化参数
    int expendSize = loadGsNum;
	auto new_xyz = torch::zeros({ expendSize,3 }).to(device);
	auto new_features_dc = torch::zeros({ expendSize,1,3 }).to(device);
	auto new_features_rest = torch::zeros({ expendSize,15,3 }).to(device);
	auto new_opacities = torch::zeros({ expendSize,1 }).to(device);
	auto new_scaling = torch::zeros({ expendSize,3 }).to(device);
	auto new_rotation = torch::zeros({ expendSize,4 }).to(device);
    std::vector<torch::Tensor>TensorList{ new_xyz ,new_features_dc ,new_features_rest ,new_opacities ,new_scaling ,new_rotation };
	int i = 0;
	for (auto& group : optimizer->param_groups()) {
		assert(group.params().size() == 1);
		auto param = group.params()[0];
		auto& state = optimizer->state();


        std::cout << state.size();
		auto& param_state = state.at(param.unsafeGetTensorImpl());
        auto d = dynamic_cast<torch::optim::AdamParamState*>(param_state.get());//param_state是一个基类，不能直接得到exp_avg，所以得dynamic_cast转换到子类
        auto stored_state = *d;
        param = TensorList[i].set_requires_grad(true);
		auto num = state.erase(group.params()[0].unsafeGetTensorImpl());
		state.insert({ (void*)param.unsafeGetTensorImpl(),std::make_unique<torch::optim::AdamParamState>(stored_state) });
        group.params()[0] = param;
		i++;
	}

	xyz = register_parameter("_xyz_"+std::to_string(step), optimizer->param_groups()[0].params()[0], true);//由于优化器的param必须是已经注册的张量，所以这里必须注册
	scaling = register_parameter("_scaling_" + std::to_string(step), optimizer->param_groups()[4].params()[0], true);//由于load时是按照注册时的名字一一对应关系读取的，所以这里注册相同的名字
	rotation = register_parameter("_ratations_" + std::to_string(step), optimizer->param_groups()[5].params()[0], true);
	opacity = register_parameter("_opacity_" + std::to_string(step), optimizer->param_groups()[3].params()[0], true);
	features_dc = register_parameter("_features_dc_" + std::to_string(step), optimizer->param_groups()[1].params()[0], true);
	features_rest = register_parameter("_features_rest_" + std::to_string(step), optimizer->param_groups()[2].params()[0], true);

}

void GaussianModel::densification_postfix(torch::Tensor new_xyz, torch::Tensor new_features_dc, torch::Tensor new_features_rest, torch::Tensor new_opacities, torch::Tensor new_scaling, torch::Tensor new_rotation) {
    std::vector<torch::Tensor>TensorList{ new_xyz ,new_features_dc ,new_features_rest ,new_opacities ,new_scaling ,new_rotation };
    std::vector<torch::Tensor>optimizable_tensors;
    optimizable_tensors = cat_tensors_to_optimizer(TensorList, optimizer.get());
    xyz = optimizable_tensors[0];
    features_dc = optimizable_tensors[1];
    features_rest = optimizable_tensors[2];
    opacity = optimizable_tensors[3];
    scaling = optimizable_tensors[4];
    rotation = optimizable_tensors[5];
    xyz_gradient_accum = torch::zeros({ get_xyz().sizes()[0], 1 }).to(device);
    denom = torch::zeros({ get_xyz().sizes()[0], 1 }).to(device);
    max_radii2D = torch::zeros({ get_xyz().sizes()[0] }).to(device);//torch::cat({max_radii2D,torch::zeros({ new_xyz.sizes()[0] }).to(device)}, 0).to(device);
}


void GaussianModel::densify_and_split(torch::Tensor grads, float grad_threshold, float scene_extent) {
    int N = 2;
    int n_init_points = get_xyz().sizes()[0];
    auto padded_grad = torch::zeros({ n_init_points }).to(device);
    padded_grad.index({slice(0,grads.sizes()[0])}) = grads.squeeze();
    auto selected_pts_mask = padded_grad.ge(grad_threshold).to(torch::kByte);
    torch::Tensor condition = std::get<0>(torch::max(get_scaling(), 1)) >= (percent_dense * scene_extent);
    selected_pts_mask = torch::logical_and(selected_pts_mask, condition).to(torch::kInt32);
    auto index = torch::nonzero(torch::where(selected_pts_mask > 0, torch::ones_like(selected_pts_mask), torch::zeros_like(selected_pts_mask))).squeeze();
    auto stds = get_scaling().index_select({ 0 }, index).repeat({ N, 1 });
    auto means = torch::zeros({ stds.sizes()[0], 3 }).to(device);
    auto samples = torch::randn_like(stds);
    samples = samples  * stds + means;
    auto rots = build_rotation(rotation.index_select({ 0 }, index)).repeat({ N, 1, 1 }).to(device);
    auto new_xyz = rots.bmm(samples.unsqueeze(-1)).squeeze(-1) + get_xyz().index_select({ 0 }, index).repeat({ N, 1 }).clone();
    auto new_scaling = scaling_inverse_activation(get_scaling().index_select({ 0 }, index).repeat({ N, 1 }) / (0.8 * N)).clone();
    auto new_rotation = rotation.index_select({ 0 }, index).repeat({ N, 1 }).clone();
    auto new_features_dc = features_dc.index_select({ 0 }, index).repeat({ N, 1, 1 }).clone();
    auto new_features_rest = features_rest.index_select({ 0 }, index).repeat({ N, 1, 1 }).clone();
    auto new_opacity = opacity.index_select({ 0 }, index).repeat({ N, 1 }).clone();

    densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation);

    auto prune_filter = torch::cat({ selected_pts_mask.to(torch::kByte), torch::zeros({N * selected_pts_mask.sum().item<int>()}, torch::kByte).to(device)}, 0);
    //剪枝
    prune_points(prune_filter);
}

std::vector<torch::Tensor> GaussianModel::_prune_optimizer(torch::Tensor mask_index) {
	std::vector<torch::Tensor> optimizable_tensors;
	int i = 0;
	for (auto& group : optimizer->param_groups()) {

		auto param = group.params()[0];


		auto& state = optimizer->state();//切记需要在step后才能得到每个param对应的OptimizerParamState，比如这个优化器里有6个参数，那么这个state表的size就为6，key为参数的unsafeGetTensorImpl，value为OptimizerParamState的uniqueptr，OptimizerParamState可以找到exp_avg等
		auto& param_state = state.at(param.unsafeGetTensorImpl());

		auto d = dynamic_cast<torch::optim::AdamParamState*>(param_state.get());//param_state是一个基类，不能直接得到exp_avg，所以得dynamic_cast转换到子类

		torch::Tensor exp_avg = d->exp_avg();
		torch::Tensor exp_avg_sq = d->exp_avg_sq();
        exp_avg = exp_avg.index_select({ 0 }, mask_index);
        exp_avg_sq = exp_avg_sq.index_select({ 0 }, mask_index);
		d->exp_avg(exp_avg);
		d->exp_avg_sq(exp_avg_sq);
		auto stored_state = *d;


        param = param.index_select({ 0 }, mask_index);
		param.set_requires_grad(true);
		optimizable_tensors.push_back(param);
		auto num = state.erase(group.params()[0].unsafeGetTensorImpl());
		state.insert({ (void*)param.unsafeGetTensorImpl(),std::make_unique<torch::optim::AdamParamState>(stored_state) });

		group.params()[0] = param;//这里的赋值操作会导致    group.params()[0] 的TensorBase改变，而当step时如果paramgroup中出现了一个新的TensorBase会自动创建一个新的paramState 加进去，所以在赋值后应该删除原始paramState，再将stored_state加进去，这样step之后就不会自动增加了
		i++;
	}
	return optimizable_tensors;
}


void GaussianModel::prune_points(torch::Tensor mask) {
    torch::Tensor valid_points_mask = torch::logical_not(mask);
    auto index = torch::nonzero(torch::where(valid_points_mask > 0, torch::ones_like(valid_points_mask), torch::zeros_like(valid_points_mask))).squeeze();
    auto optimizable_tensors = _prune_optimizer(index);
	xyz = optimizable_tensors[0];
	features_dc = optimizable_tensors[1];
	features_rest = optimizable_tensors[2];
	opacity = optimizable_tensors[3];
	scaling = optimizable_tensors[4];
	rotation = optimizable_tensors[5];

    xyz_gradient_accum = xyz_gradient_accum.index_select({ 0 }, index);
    denom = denom.index_select({ 0 }, index);
    max_radii2D = max_radii2D.index_select({ 0 }, index);
}


void GaussianModel::reset_opacity() {
    auto opacities_new = inverse_sigmoid(torch::min(get_opacity(), torch::ones_like(get_opacity()) * 0.01));
    auto optimizable_tensor = replace_opacity_to_optimizer(opacities_new);
    opacity = optimizable_tensor;
}


torch::Tensor GaussianModel::replace_opacity_to_optimizer(torch::Tensor tensor) {
    torch::Tensor optimizable_tensor;
    auto& group = optimizer->param_groups()[3];
	auto param = group.params()[0];


	auto& state = optimizer->state();//切记需要在step后才能得到每个param对应的OptimizerParamState，比如这个优化器里有6个参数，那么这个state表的size就为6，key为参数的unsafeGetTensorImpl，value为OptimizerParamState的uniqueptr，OptimizerParamState可以找到exp_avg等
	auto& param_state = state.at(param.unsafeGetTensorImpl());

	auto d = dynamic_cast<torch::optim::AdamParamState*>(param_state.get());//param_state是一个基类，不能直接得到exp_avg，所以得dynamic_cast转换到子类

	torch::Tensor exp_avg = d->exp_avg();
	torch::Tensor exp_avg_sq = d->exp_avg_sq();
    exp_avg = torch::zeros_like(tensor);
    exp_avg_sq = torch::zeros_like(tensor);
	d->exp_avg(exp_avg);
	d->exp_avg_sq(exp_avg_sq);
	auto stored_state = *d;

    param = tensor;
	param.set_requires_grad(true);
	auto num = state.erase(group.params()[0].unsafeGetTensorImpl());
	state.insert({ (void*)param.unsafeGetTensorImpl(),std::make_unique<torch::optim::AdamParamState>(stored_state) });

	group.params()[0] = param;
    return param;
}


void GaussianModel::saveCkpt(std::string base, std::string expname, int global_step, int epoch) {//由于save参数时只会save注册参数时的大小，所以这里再注册一次,不然只会读到注册时的大小
	xyz = register_parameter("_xyz_" + std::to_string(global_step), xyz, true);
	scaling = register_parameter("_scaling_" + std::to_string(global_step), scaling, true);
	rotation = register_parameter("_ratations_" + std::to_string(global_step), rotation, true);
	opacity = register_parameter("_opacity_" + std::to_string(global_step), opacity, true);
	features_dc = register_parameter("_features_dc_" + std::to_string(global_step), features_dc, true);
	features_rest = register_parameter("_features_rest_" + std::to_string(global_step), features_rest, true);




    std::ostringstream path_stream_model;
    path_stream_model << base << "/" << expname << "/" << "ckpts" << "/" << "gs" << "/" << std::setfill('0') << std::setw(8) << global_step << ".tar";
    std::string path_model = path_stream_model.str();

    torch::serialize::OutputArchive output_model_archive;
    this->to(torch::kCPU);
    this->save(output_model_archive);
    output_model_archive.save_to(path_model);
    this->to(device);


	std::ostringstream path_stream_opt;
	path_stream_opt << base << "/" << expname << "/" << "ckpts" << "/" << "optimizer" << "/" << std::setfill('0') << std::setw(8) << global_step << ".tar";
	std::string path_opt = path_stream_opt.str();

	torch::serialize::OutputArchive output_optim_archive;
	optimizer->save(output_optim_archive);
	torch::Tensor global_step_tensor = torch::tensor({ global_step }, torch::kInt);
	torch::Tensor epoch_tensor = torch::tensor({ epoch }, torch::kInt);
	torch::Tensor gsNum_tensor = torch::tensor({ xyz.sizes()[0]}, torch::kInt);
	output_optim_archive.write("global_step", global_step_tensor);
	output_optim_archive.write("epoch", epoch_tensor);
	output_optim_archive.write("gsNum", gsNum_tensor);
	output_optim_archive.save_to(path_opt);
}
namespace fs = std::filesystem;
std::string GaussianModel::find_max_tar_file(const std::string& basedir, const std::string& expname, const std::string& objName) {
	std::string path = basedir + expname + "\\" + "ckpts" + "\\" + objName;
	std::vector<int> versions;

	for (const auto& entry : fs::directory_iterator(path)) {
		if (entry.is_regular_file() && entry.path().extension() == ".tar") {
			// 假设文件名格式为 "xxxx.tar"，提取数字部分
			std::string filename = entry.path().filename().string();
			std::string version = filename.substr(0, filename.size() - 4);
			try {
				int num = std::stoi(version);
				versions.push_back(num);
			}
			catch (const std::invalid_argument& e) {
				// 如果转换失败，忽略这个文件
				continue;
			}
		}
	}

	// 如果找到了版本号，返回最大的那个
	if (!versions.empty()) {
		auto max_version = *std::max_element(versions.begin(), versions.end());
		std::stringstream ss;
		ss << std::setw(8) << std::setfill('0') << max_version;
		std::string max_file = ss.str() + ".tar";
		return (fs::path(path) / max_file).string();
	}
	else {
		return "";
	}
}
void GaussianModel::loadCkpt(std::string base, std::string expname, int* global_step, int* epoch) {//加载load数据其实本质是赋值，所以必须得先把高斯模型的张量和优化器的张量大小扩大为load数据大小

    auto gsPath = find_max_tar_file(base, expname, "gs");
    auto optimizerPath = find_max_tar_file(base, expname, "optimizer");
    if ((gsPath == "") || (optimizerPath == ""))
        return;
    std::cout << "找到了检查点！正在加载" << std::endl;
    
	OptimizationParameters op;
	training_setup(op);

    int gsNum;
	torch::serialize::InputArchive input_archive2;
	input_archive2.load_from(optimizerPath);
	// 读取额外信息
	torch::Tensor global_step_tensor;
	torch::Tensor epoch_tensor;
	torch::Tensor gsNum_tensor;
	input_archive2.read("global_step", global_step_tensor);
	input_archive2.read("epoch", epoch_tensor);
	input_archive2.read("gsNum", gsNum_tensor);

	// 将Tensor转换为int
	*global_step = global_step_tensor.item<int>();
	*epoch = epoch_tensor.item<int>();
	gsNum = gsNum_tensor.item<int>();

	// Load optim state
	optimizer->load(input_archive2);

    //将模型和优化器的张量扩展
    expend_gs_and_optimizer(gsNum,*global_step);



    // Load model state
    torch::serialize::InputArchive input_archive;
    input_archive.load_from(gsPath);
    this->load(input_archive);
    this->to(device);

	percent_dense = op.percent_dense;
	xyz_gradient_accum = torch::zeros({ get_xyz().sizes()[0], 1 }).to(device);
	denom = torch::zeros({ get_xyz().sizes()[0], 1 }).to(device);
    max_radii2D = torch::zeros({ get_xyz().sizes()[0] }).to(device);
}


float GaussianModel::get_lr_from_iteration(int step,float lr_init,float lr_final) {
    float lr_delay_steps = 0;
    float lr_delay_mult = 1.0f;
    int max_steps = 1000000;
    float delay_rate;
    if (step < 0 || (lr_init == 0.0 && lr_final == 0.0))
        return 0.0;
    if (lr_delay_steps > 0) {
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * sin(0.5 * PI * std::min(std::max((float)step / lr_delay_steps, 0.f), 1.f));
    }
    else
        delay_rate = 1.0;
    auto t = std::min(std::max((float)step / (float)max_steps, 0.f), 1.f);
    float log_lerp = std::exp(std::log(lr_init) * (1 - t) + std::log(lr_final) * t);
    return delay_rate * log_lerp;
}

void GaussianModel::update_lr_xyz(int step) {
    float new_lr = get_lr_from_iteration(step,spatial_lr_scale*position_lr_init,spatial_lr_scale*position_lr_final);
    //std::cout <<"新lr" << new_lr << std::endl;
    optimizer->param_groups()[0].options().set_lr(double(new_lr));
}