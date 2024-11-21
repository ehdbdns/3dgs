#pragma once
#include"util.h"
#include"ParameterConfig.h"
#include"GaussianModel.h"


class Camera {
public:
	int uid;
	torch::Tensor R; 
	torch::Tensor T;
	float FoVx;
	float FoVy;
	std::string image_name;
	torch::Tensor original_image;
	int image_width;
	int image_height;
	torch::Tensor gt_alpha_mask;  
	float zfar;
	float znear;
	torch::Tensor trans;
	float scale;
	torch::DeviceType data_device;
	torch::Tensor world_view_transform;
	torch::Tensor projection_matrix;
	torch::Tensor full_proj_transform;
	torch::Tensor camera_center;
	int imgID;
	cv::Mat image;

public:
	Camera(int colmap_id, torch::Tensor R, torch::Tensor T, float FoVx, float FoVy,
		torch::Tensor image, torch::Tensor alpha_mask, std::string img_name);
	Camera(const Camera& other);
};


// 定义CameraModel结构体
struct CameraModel {
	CameraModel() = default;
	int model_id;
	std::string model_name;
	int num_params;
};

// 定义Camera结构体
struct CameraBin {
	int id;
	std::string model;
	int width;
	int height;
	std::vector<double> params;
};

// 定义Image结构体
struct ImageBin {
	int id;
	std::vector<double> qvec;
	std::vector<double> tvec;
	int camera_id;
	std::string name;
	std::vector<std::pair<double, double>> xys;
	std::vector<int> point3D_ids;
};

// 定义Point3D结构体
struct Point3D {
	int id;
	std::vector<double> xyz;
	std::vector<double> rgb;
	double error;
	std::vector<int> image_ids;
	std::vector<int> point2D_idxs;
};






class Scene {
public:
	Scene( ModelParameters& mp,GaussianModel& gs,bool ckpt);
	void Scene::updateViewCamera(char viewTrans, Camera* viewCamera, Camera* resetCam, int deltaX, int deltaY);

	std::string model_path;
	GaussianModel* gsModel;
	std::vector<cameraPose>cameraposes;
	std::vector<Camera>Cameras;
	torch::Tensor cameras_center_avg;
	float camera_extent;
	torch::Tensor cameras_upsum;
	// 创建CAMERA_MODELS数组
	std::vector<CameraModel> CAMERA_MODELS;

	// 创建CAMERA_MODEL_IDS映射
	std::map<int, CameraModel> CAMERA_MODEL_IDS;
	// 创建CAMERA_MODEL_NAMES映射
	std::map<std::string, CameraModel> CAMERA_MODEL_NAMES;

	std::stack<int> indexStack;

	int cameIndex = 0;

	// 初始化映射
	void initialize_camera_models();

	void read_extrinsics_binary(const std::string& path_to_model_file, std::vector<ImageBin>& imagebins);
	void read_intrinsics_binary(const std::string& path_to_model_file, std::vector<CameraBin>& camerabins);
	Camera getRandomCamera();
	torch::Tensor rotateX(float theta);
	torch::Tensor rotateY(float theta);

};



std::vector<cameraPose> loadLLFFData(std::string FileName, std::string imgBaseName, std::string imgExtension);
void read_extrinsics_binary(const std::string& path_to_model_file, std::vector<Camera>& cameras);
uint64_t read_next_bytes(std::ifstream& fid, size_t num_bytes);
void read_intrinsics_binary(const std::string& path_to_model_file);
std::vector<Camera> readColmapCameras(const std::vector<ImageBin>& cam_extrinsics,const std::vector<CameraBin>& cam_intrinsics,const std::string& images_folder);
torch::Tensor poses_avg(std::vector<Camera>& cameras, torch::Tensor* upsum_out);