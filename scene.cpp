#include"scene.h"



std::vector<cameraPose> loadLLFFData(std::string FileName, std::string imgBaseName, std::string imgExtension) {

	cnpy::NpyArray poses_arr = cnpy::npy_load(FileName);
	std::cout << poses_arr.num_vals << " " << poses_arr.shape[0] << " " << poses_arr.shape[1] << " " << poses_arr.word_size << " " << poses_arr.fortran_order;
	std::vector<double>arr_vec = poses_arr.as_vec<double>();//读取到的顺序：r r r t1 H r r r t2 W r r r t3 f
	int poseNum = int(arr_vec.size()) / 17;
	//将读取到的信息存进结构体
	std::vector<cameraPose>cameraposes(poseNum);//(poses_arr.shape[0]);
	for (int i = 0;i < poseNum;i++) {
		cameraposes[i].pose = torch::from_blob(arr_vec.data() + 17 * i, { 3,5 }, torch::kDouble);//T (3,5)
		cameraposes[i].pose = torch::cat({ cameraposes[i].pose.index({torch::indexing::Slice(),torch::indexing::Slice(1,2)}),//这里将读来的相机位姿从LLFF坐标系转换到右手系
			-cameraposes[i].pose.index({torch::indexing::Slice(),torch::indexing::Slice(0,1)}) ,cameraposes[i].pose.index({torch::indexing::Slice(),torch::indexing::Slice(2,3)}),
			cameraposes[i].pose.index({torch::indexing::Slice(),torch::indexing::Slice(3,4)}) ,cameraposes[i].pose.index({torch::indexing::Slice(),torch::indexing::Slice(4,5)}) }, 1);//T (3,5)
		cameraposes[i].cameraImg = cv::imread(imgBaseName + " (" + std::to_string(i + 1) + ")" + imgExtension);
		//print(cameraposes[i].pose);
	}
	return cameraposes;
}
void Scene::initialize_camera_models() {
	CAMERA_MODELS = {
	{0, std::string("SIMPLE_PINHOLE"), 3},
	{1, "PINHOLE", 4},
	{2, "SIMPLE_RADIAL", 4},
	{3, "RADIAL", 5},
	{4, "OPENCV", 8},
	{5, "OPENCV_FISHEYE", 8},
	{6, "FULL_OPENCV", 12},
	{7, "FOV", 5},
	{8, "SIMPLE_RADIAL_FISHEYE", 4},
	{9, "RADIAL_FISHEYE", 5},
	{10, "THIN_PRISM_FISHEYE", 12}
	};
	for (const auto& model : CAMERA_MODELS) {
		CAMERA_MODEL_IDS[model.model_id] = model;
		CAMERA_MODEL_NAMES[model.model_name] = model;
	}
}

uint64_t read_next_bytes(std::ifstream& fid, size_t num_bytes) {
	uint64_t value = 0;
	fid.read(reinterpret_cast<char*>(&value), num_bytes);
	if (!fid) {
		throw std::runtime_error("Error reading from file");
	}
	return value;
}

void Scene::read_extrinsics_binary(const std::string& path_to_model_file,std::vector<ImageBin>&imagebins) {//这里colmap出错了，
	std::ifstream fid(path_to_model_file, std::ios::binary);
	if (!fid) {
		throw std::runtime_error("Unable to open file");
	}

	uint64_t num_reg_images = read_next_bytes(fid, sizeof(uint64_t));
	imagebins.resize(num_reg_images);
	for (uint64_t i = 0; i < num_reg_images; ++i) {
		int image_id;
		double qvec[4];
		double tvec[3];
		int camera_id;
		uint64_t num_points2D;

		fid.read(reinterpret_cast<char*>(&image_id), sizeof(image_id));
		fid.read(reinterpret_cast<char*>(qvec), sizeof(qvec));
		fid.read(reinterpret_cast<char*>(tvec), sizeof(tvec));
		fid.read(reinterpret_cast<char*>(&camera_id), sizeof(camera_id));

		std::string image_name;
		char current_char;
		do {
			fid.read(&current_char, 1);
			if (current_char != '\x00') {
				image_name += current_char;
			}
		} while (current_char != '\x00');

		num_points2D = read_next_bytes(fid, sizeof(uint64_t));

		std::vector<double> x_y_id_s(num_points2D * 3);
		fid.read(reinterpret_cast<char*>(x_y_id_s.data()), num_points2D * 3 * sizeof(double));

		std::vector<std::pair<double, double>> xys;
		std::vector<int> point3D_ids;
		for (uint64_t j = 0; j < num_points2D; ++j) {
			xys.emplace_back(x_y_id_s[j * 3], x_y_id_s[j * 3 + 1]);
			point3D_ids.push_back(static_cast<int>(x_y_id_s[j * 3 + 2]));
		}
		ImageBin im;
		im.camera_id = camera_id;
		im.id = image_id;
		im.name = image_name;
		im.point3D_ids = point3D_ids;
		im.qvec = std::vector<double>(std::begin(qvec), std::end(qvec));
		im.tvec = std::vector<double>(std::begin(tvec), std::end(tvec));
		im.xys = xys;
		imagebins[image_id-1]=im;
	}
}
void Scene::read_intrinsics_binary(const std::string& path_to_model_file, std::vector<CameraBin>& camerabins) {
	std::ifstream fid(path_to_model_file, std::ios::binary);
	if (!fid) {
		throw std::runtime_error("Unable to open file");
	}

	uint64_t num_cameras = read_next_bytes(fid, sizeof(uint64_t));

	for (uint64_t i = 0; i < num_cameras; ++i) {
		int camera_id;
		int model_id;
		uint64_t width;
		uint64_t height;
		fid.read(reinterpret_cast<char*>(&camera_id), sizeof(camera_id));
		fid.read(reinterpret_cast<char*>(&model_id), sizeof(model_id));
		fid.read(reinterpret_cast<char*>(&width), sizeof(width));
		fid.read(reinterpret_cast<char*>(&height), sizeof(height));

		// 假设CAMERA_MODEL_IDS是一个全局的map，你需要根据实际情况来定义它
		std::string model_name = CAMERA_MODEL_IDS[model_id].model_name;
		int num_params = CAMERA_MODEL_IDS[model_id].num_params;

		std::vector<double> params(num_params);
		fid.read(reinterpret_cast<char*>(params.data()), num_params * sizeof(double));
		CameraBin ca;
		ca.height = height;
		ca.width = width;
		ca.id = camera_id;
		ca.model = model_name;
		ca.params = params;
		camerabins.push_back(ca);
	}
}

std::vector<Camera> readColmapCameras(const std::vector<ImageBin>& cam_extrinsics,const std::vector<CameraBin>& cam_intrinsics,const std::string& images_folder) {
	std::vector<Camera> cam_infos;
	for (size_t idx = 0; idx < cam_extrinsics.size(); ++idx) {
		std::cout << "\rReading camera " << idx + 1 << "/" << cam_extrinsics.size() << std::flush;

		const auto& extr = cam_extrinsics[idx];
		const auto& intr = cam_intrinsics[extr.camera_id-1];
		int height = intr.height;
		int width = intr.width;

		int uid = intr.id;
		torch::Tensor R = qvec2rotmat(torch::tensor({ extr.qvec[0],extr.qvec[1] ,extr.qvec[2],extr.qvec[3] })).permute({1,0});
		torch::Tensor T = torch::tensor({ extr.tvec[0], extr.tvec[1], extr.tvec[2] }).unsqueeze(1);
		double FovY, FovX;
		if (intr.model == "SIMPLE_PINHOLE") {
			double focal_length_x = intr.params[0];
			FovY = focal2fov(focal_length_x, height);
			FovX = focal2fov(focal_length_x, width);
		}
		else if (intr.model == "PINHOLE") {
			double focal_length_x = intr.params[0];
			double focal_length_y = intr.params[1];
			FovY = focal2fov(focal_length_y, height);
			FovX = focal2fov(focal_length_x, width);
		}
		else {
			throw std::runtime_error("Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!");
		}


		std::string image_path = (std::filesystem::path(images_folder) / extr.name).string();
		std::string image_name = std::filesystem::path(extr.name).stem().string();
		cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
		torch::Tensor imgTensor = torch::from_blob(image.data, { image.rows,image.cols, 3 }, torch::kByte).div(255.0).to(torch::kFloat32);//H,W,3
		imgTensor = torch::cat({ imgTensor.index({torch::indexing::Slice(),torch::indexing::Slice(),torch::indexing::Slice(2,3)}),
		imgTensor.index({torch::indexing::Slice(),torch::indexing::Slice(),torch::indexing::Slice(1,2)}),
		imgTensor.index({torch::indexing::Slice(),torch::indexing::Slice(),torch::indexing::Slice(0,1)}) }, 2);//因为读取的是BGR图片，转换成RGB
		torch::Tensor gt_alpha_mas;
		Camera c(idx, R, T, FovX, FovY, imgTensor, gt_alpha_mas, image_name);
		c.imgID = idx;
		c.image_name = image_name;
		c.image = image;
		cam_infos.push_back(c);
	}
	std::cout << '\n';
	return cam_infos;
}


void get_center_and_diag(std::vector<torch::Tensor> camera_centers, torch::Tensor& center, float* diagonal) {
	auto camera_centers_stack = torch::cat(camera_centers,1);
	auto avg_cam_center = torch::mean(camera_centers_stack, 1, true);
	center = avg_cam_center;
	auto dist = torch::norm(camera_centers_stack - center, 2, 0,true );
	*diagonal = torch::max(dist).item<float>();
}
void getNerfppNorm(std::vector<Camera>& cameras, torch::Tensor& center, float* extent) {
	std::vector<torch::Tensor>cam_centers;
	for (int i = 0;i < cameras.size();i++) {
		auto c = cameras[i];
		auto W2C = getWorld2View2(c.R, c.T);
		auto C2W = torch::inverse(W2C);
		cam_centers.push_back(C2W.index({slice(0,3),slice(3,4),}));
	}

	float diagonal;
	get_center_and_diag(cam_centers, center, &diagonal);//center是所有相机的平均位置，radii是所有相机的包围盒extent
	float radius = diagonal * 1.1;
	*extent = radius;
}

torch::Tensor poses_avg(std::vector<Camera>& cameras, torch::Tensor* upsum_out) {
	//计算所有相机的平均位置和平均坐标系朝向
	torch::Tensor center = torch::zeros({ 3,1 });
	torch::Tensor vec2 = torch::zeros({ 3,1 });
	torch::Tensor up = torch::zeros({ 3,1 });//T
	for (int i = 0;i < cameras.size();i++) {
		center += cameras[i].T;
		vec2 += cameras[i].R.index({ torch::indexing::Slice(0,3),torch::indexing::Slice(2,3) });
		up += cameras[i].R.index({ torch::indexing::Slice(0,3),torch::indexing::Slice(1,2) });
	}
	center /= cameras.size();
	vec2 /= torch::norm(vec2, 2);//normalize(vec2)//T//平均朝向
	if (upsum_out != nullptr)
		*upsum_out = up;
	return *upsum_out;
}


Camera::Camera(int colmap_id, torch::Tensor R, torch::Tensor T, float FoVx, float FoVy,
	torch::Tensor image, torch::Tensor alpha_mask, std::string img_name) {
	this->R = R;
	this->T = T;
	this->FoVx = FoVx;
	this->FoVy = FoVy;
	this->image_name = img_name;
	this->gt_alpha_mask = alpha_mask;

	this->data_device = device;



	this->original_image = image.clamp(0.0, 1.0).to(this->data_device);
	this->image_width = this->original_image.size(1);
	this->image_height = this->original_image.size(0);

	if (gt_alpha_mask.defined()) {
		this->original_image *= gt_alpha_mask.to(this->data_device);
	}
	else {
		this->original_image *= torch::ones({this->image_height, this->image_width,1 }, this->data_device);
	}

	this->zfar = 100.0;
	this->znear = 0.01;

	this->trans = torch::tensor({0,0,0});
	this->scale = 1.0f;


	this->world_view_transform = getWorld2View2(R, T).transpose(0, 1).to(this->data_device);
	this->projection_matrix = getProjectionMatrix(this->znear, this->zfar, this->FoVx, this->FoVy).transpose(0, 1).to(this->data_device);
	this->full_proj_transform = this->world_view_transform.unsqueeze(0).bmm(this->projection_matrix.unsqueeze(0)).squeeze(0);
	this->camera_center = torch::inverse(this->world_view_transform).index({ slice(3,4),slice(0,3) }).to(this->data_device);
}
Camera::Camera(const Camera& other) {
	this->T = other.T.clone();
	this->R = other.R.clone();
	FoVx=other.FoVx;
	FoVy=other.FoVy;
	original_image=other.original_image;
	image_width=other.image_width;
	image_height=other.image_height;
	gt_alpha_mask=other.gt_alpha_mask;
	zfar=other.zfar;
	znear=other.znear;
	trans=other.trans;
	scale=other.scale;
	data_device=other.data_device;
	world_view_transform=other.world_view_transform;
	projection_matrix=other.projection_matrix;
	full_proj_transform=other.full_proj_transform;
	camera_center=other.camera_center;
	image_name = other.image_name;
	image = other.image;
}

Scene::Scene(ModelParameters& mp,GaussianModel& gs,bool ckpt) {

	initialize_camera_models();

	model_path = mp._model_path;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	if (pcl::io::loadPLYFile("C:\\mytinyrenderproj\\MyRenderer\\3dgs\\points3D.ply", *cloud) == -1) {
		abort();
	}
	//pcl::visualization::CloudViewer viewer("Cloud Viewer");
	//viewer.showCloud(cloud);
	//system("pause");

	if(!ckpt)
		gs.createFromPcd(cloud, 0.9);


	gsModel = &gs;

	std::vector<CameraBin> camerabins;
	std::vector<ImageBin> imagebins;
	read_extrinsics_binary("C:\\Users\\86155\\Desktop\\gaussian-splatting\\data\\sparse\\0\\images.bin", imagebins);
	read_intrinsics_binary("C:\\Users\\86155\\Desktop\\gaussian-splatting\\data\\sparse\\0\\cameras.bin",camerabins);

	Cameras = readColmapCameras(imagebins, camerabins, "C:\\Users\\86155\\Desktop\\gaussian-splatting\\data\\images");

	auto poseAvg = poses_avg(Cameras, &cameras_upsum);

	cameras_upsum /= torch::norm(cameras_upsum, 2);

	getNerfppNorm(Cameras, cameras_center_avg, &camera_extent);

	gs.spatial_lr_scale = camera_extent;

	//初始化相机栈
	std::random_device rd;
	std::mt19937 g(rd());
	std::vector<int>indices;
	for (int i = 0; i < Cameras.size(); ++i) {
		indices.push_back(i);
	}
	std::shuffle(indices.begin(), indices.end(), g);
	for (int i = 0; i < Cameras.size(); ++i) {
		indexStack.push(indices[i]);
	}
}


torch::Tensor Scene::rotateX(float theta) {
	auto theta_rad = theta * PI / 180; // 将角度转换为弧度
	double cos_theta = std::cos(theta_rad);
	double sin_theta = std::sin(theta_rad);

	// 构建绕X轴旋转的矩阵
	torch::Tensor R_x = torch::tensor({
		{1., 0., 0.},
		{0., cos_theta, -sin_theta},
		{0., sin_theta, cos_theta}
		});

	return R_x;
}

// 绕Y轴旋转theta角度
torch::Tensor Scene::rotateY(float theta) {
	auto theta_rad = theta * PI / 180; // 将角度转换为弧度
	double cos_theta = std::cos(theta_rad);
	double sin_theta = std::sin(theta_rad);

	// 构建绕Y轴旋转的矩阵
	torch::Tensor R_y = torch::tensor({
		{cos_theta, 0., sin_theta},
		{0., 1., 0.},
		{-sin_theta, 0., cos_theta}
		});

	return R_y;
}



void Scene::updateViewCamera(char viewTrans, Camera* viewCamera,Camera*resetCam,int deltaX,int deltaY) {
	int rotSpeed = 5;
	auto W2C = getWorld2ViewCamera2(viewCamera->R, viewCamera->T);
	auto C2W = torch::inverse(W2C);
	auto toward = C2W.index({slice(0,3),slice(2,3)});
	auto right = C2W.index({ slice(0,3),slice(0,1) });
	auto rotxMat = rotateY(rotSpeed*deltaX);
	auto rotyMat = rotateX(rotSpeed*deltaY);
	auto rotMat = rotxMat.matmul(rotyMat);
	viewCamera->R =viewCamera->R.matmul(rotMat);
	switch(viewTrans){
	case '2': {
		viewCamera->T += 0.1*right;
		break;
	}
	case '3': {
		viewCamera->T -= 0.1*right;
		break;
	}
	case '4': {
		viewCamera->T -= 0.1*toward;
		break;
	}
	case '5': {
		viewCamera->T += 0.1*toward;
		break;
	}
	case '6': {

		viewCamera->T = Cameras[cameIndex].T.clone();
		viewCamera->R = Cameras[cameIndex].R.clone();
		std::cout << Cameras[cameIndex].image_name << std::endl;
		cameIndex++;
		if (cameIndex >= Cameras.size())
			cameIndex = 0;
		viewCamera->world_view_transform = getWorld2View2(viewCamera->R, viewCamera->T).transpose(0, 1).to(viewCamera->data_device);
		viewCamera->projection_matrix = getProjectionMatrix(viewCamera->znear, viewCamera->zfar, viewCamera->FoVx, viewCamera->FoVy).transpose(0, 1).to(viewCamera->data_device);
		viewCamera->full_proj_transform = viewCamera->world_view_transform.unsqueeze(0).bmm(viewCamera->projection_matrix.unsqueeze(0)).squeeze(0);
		viewCamera->camera_center = torch::inverse(viewCamera->world_view_transform).index({ slice(3,4),slice(0,3) }).to(viewCamera->data_device);
		return;
		break;
	}
	case '7': {

		viewCamera->T = Cameras[cameIndex].T.clone();
		viewCamera->R = Cameras[cameIndex].R.clone();
		std::cout << Cameras[cameIndex].image_name << std::endl;
		cameIndex--;
		if (cameIndex < 0)
			cameIndex = Cameras.size()-1;
		viewCamera->world_view_transform = getWorld2View2(viewCamera->R, viewCamera->T).transpose(0, 1).to(viewCamera->data_device);
		viewCamera->projection_matrix = getProjectionMatrix(viewCamera->znear, viewCamera->zfar, viewCamera->FoVx, viewCamera->FoVy).transpose(0, 1).to(viewCamera->data_device);
		viewCamera->full_proj_transform = viewCamera->world_view_transform.unsqueeze(0).bmm(viewCamera->projection_matrix.unsqueeze(0)).squeeze(0);
		viewCamera->camera_center = torch::inverse(viewCamera->world_view_transform).index({ slice(3,4),slice(0,3) }).to(viewCamera->data_device);
		return;
		break;
	}
	}

	//if (viewTrans != '1') {
	viewCamera->world_view_transform = getWorld2ViewCamera(viewCamera->R, viewCamera->T).transpose(0, 1).to(viewCamera->data_device);
	viewCamera->projection_matrix = getProjectionMatrix(viewCamera->znear, viewCamera->zfar, viewCamera->FoVx, viewCamera->FoVy).transpose(0, 1).to(viewCamera->data_device);
	viewCamera->full_proj_transform = viewCamera->world_view_transform.unsqueeze(0).bmm(viewCamera->projection_matrix.unsqueeze(0)).squeeze(0);
	viewCamera->camera_center = torch::inverse(viewCamera->world_view_transform).index({ slice(3,4),slice(0,3) }).to(viewCamera->data_device);
	//}
}


Camera Scene::getRandomCamera() {
	if (indexStack.empty()) {
		std::random_device rd;
		std::mt19937 g(rd());
		std::vector<int>indices;
		for (int i = 0; i < Cameras.size(); ++i) {
			indices.push_back(i);
		}
		std::shuffle(indices.begin(), indices.end(), g);
		for (int i = 0; i < Cameras.size(); ++i) {
			indexStack.push(indices[i]);
		}
	}
	int index = indexStack.top();
	indexStack.pop();
	Camera c = Cameras[index];
	return c;
}