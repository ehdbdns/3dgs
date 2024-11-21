#pragma once
#include"util.h"


torch::Tensor inverse_sigmoid(torch::Tensor x) {
    return torch::log(x / (1 - x));
}


torch::Tensor normalize(torch::Tensor x) {
    return x / torch::norm(x, 2);
}


torch::Tensor build_rotation(torch::Tensor r) {//从四元数构建旋转矩阵
    auto norm = torch::sqrt(r.index({ slice(),slice(0,1) }) * r.index({ slice(),slice(0,1) }) + r.index({ slice(),slice(1,2) }) * r.index({ slice(),slice(1,2) })
        + r.index({ slice(),slice(2,3) }) * r.index({ slice(),slice(2,3) }) + r.index({ slice(),slice(3,4) }) * r.index({ slice(),slice(3,4) }));

    auto q = r / norm.index({ slice(),slice(torch::indexing::None) });

    auto R = torch::zeros({ q.sizes()[0], 3, 3 }).to(device);

    r = q.index({ slice(),slice(0,1) });
    auto x = q.index({ slice(),slice(1,2) });
    auto y = q.index({ slice(),slice(2,3) });
    auto z = q.index({ slice(),slice(3,4) });

    R.index({ slice(),slice(0,1),slice(0,1) }) = 1 - 2 * (y * y + z * z).unsqueeze(2);
    R.index({ slice(),slice(0,1),slice(1,2) }) = 2 * (x * y - r * z).unsqueeze(2);
    R.index({ slice(),slice(0,1),slice(2,3) }) = 2 * (x * z + r * y).unsqueeze(2);
    R.index({ slice(),slice(1,2),slice(0,1) }) = 2 * (x * y + r * z).unsqueeze(2);
    R.index({ slice(),slice(1,2),slice(1,2) }) = 1 - 2 * (x * x + z * z).unsqueeze(2);
    R.index({ slice(),slice(1,2),slice(2,3) }) = 2 * (y * z - r * x).unsqueeze(2);
    R.index({ slice(),slice(2,3),slice(0,1) }) = 2 * (x * z - r * y).unsqueeze(2);
    R.index({ slice(),slice(2,3),slice(1,2) }) = 2 * (y * z + r * x).unsqueeze(2);
    R.index({ slice(),slice(2,3),slice(2,3) }) = 1 - 2 * (x * x + y * y).unsqueeze(2);
    return R;
}


torch::Tensor build_scaling_rotation(torch::Tensor s, torch::Tensor r) {
    auto L = torch::zeros((s.sizes()[0], 3, 3), torch::kFloat32).to(device);
    auto R = build_rotation(r);

    L.index({ slice(),slice(0),slice(0) }) = s.index({ slice(),slice(0) });
    L.index({ slice(),slice(1),slice(1) }) = s.index({ slice(),slice(1) });
    L.index({ slice(),slice(2),slice(2) }) = s.index({ slice(),slice(2) });

    L = torch::matmul(R, L);//将旋转矩阵与缩放矩阵相乘得到scaling_rotation矩阵
    return L;
}


torch::Tensor strip_lowerdiag(torch::Tensor L) {//提取矩阵上三角部分并存进一个大小为6的张量，因为矩阵是对称的，所以只需要优化上半部分
    auto uncertainty = torch::zeros((L.sizes()[0], 6), torch::kFloat32).to(device);
    uncertainty.index({ slice(),slice(0) }) = L.index({ slice(),slice(0),slice(0) });
    uncertainty.index({ slice(),slice(1) }) = L.index({ slice(),slice(0),slice(1) });
    uncertainty.index({ slice(),slice(2) }) = L.index({ slice(),slice(0),slice(2) });
    uncertainty.index({ slice(),slice(3) }) = L.index({ slice(),slice(1),slice(1) });
    uncertainty.index({ slice(),slice(4) }) = L.index({ slice(),slice(1),slice(2) });
    uncertainty.index({ slice(),slice(5) }) = L.index({ slice(),slice(2),slice(2) });
    return uncertainty;
}


torch::Tensor build_covariance_from_scaling_rotation(torch::Tensor scaling, float scaling_modifier, torch::Tensor rotation) {//传入四元数以及缩放系数，得到该高斯的协方差矩阵的编码
    auto L = build_scaling_rotation(scaling_modifier * scaling, rotation);
    auto actual_covariance = torch::matmul(L, L.transpose(1, 2));
    auto symm = strip_lowerdiag(actual_covariance);
    return symm;
}


void print(std::string desc, torch::Tensor t)
{
	std::cout << desc << t << std::endl;
}


void print_size(torch::Tensor t) {
	std::cout << t.sizes() << std::endl;
}


torch::Tensor getWorld2View2(torch::Tensor R, torch::Tensor t) {
	torch::Tensor translate = torch::tensor({ .0,.0,.0 });
	float scale = 1.0f;
	torch::Tensor Rt = torch::zeros({ 4, 4 });

	Rt.index({ slice(0,3),slice(0,3) }) = R.transpose(0, 1);
	Rt.index({ slice(3,4),slice(3,4) }) = 1.0;

	t = t.reshape({ 3, 1 });
	Rt.index({ slice(0,3),slice(3,4) }) = t;

	torch::Tensor C2W = torch::inverse(Rt);
	torch::Tensor cam_center = C2W.index({ slice(0,3),slice(3,4) });
	cam_center = (cam_center + translate.unsqueeze(1)) * scale;
	C2W.index({ slice(0,3),slice(3,4) }) = cam_center;
	Rt = torch::inverse(C2W);

	return Rt.to(torch::kFloat32);
}

torch::Tensor getWorld2ViewCamera(torch::Tensor R,torch::Tensor t) {
    torch::Tensor translate = torch::tensor({ .0,.0,.0 });
    float scale = 1.0f;
    torch::Tensor Rt = torch::zeros({ 4, 4 });

	Rt.index({ slice(0,3),slice(0,3) }) = R.transpose(0, 1);
	Rt.index({ slice(3,4),slice(3,4) }) = 1.0;

	t = t.reshape({ 3, 1 });
    torch::Tensor T = torch::eye(4);
    T.index({slice(0,3),slice(3,4)}) = t;

    torch::Tensor C2W = torch::inverse(Rt);
    torch::Tensor cam_center = C2W.index({slice(0,3),slice(3,4)});
    cam_center = (cam_center + translate.unsqueeze(1)) * scale;
    C2W.index({slice(0,3),slice(3,4)}) = cam_center;
    Rt = torch::inverse(C2W);
    Rt = Rt.matmul(T);
    return Rt.to(torch::kFloat32);
}
torch::Tensor getWorld2ViewCamera2(torch::Tensor R, torch::Tensor t) {
	torch::Tensor translate = torch::tensor({ .0,.0,.0 });
	float scale = 1.0f;
	torch::Tensor Rt = torch::zeros({ 4, 4 });

	Rt.index({ slice(0,3),slice(0,3) }) = R.transpose(0, 1);
	Rt.index({ slice(3,4),slice(3,4) }) = 1.0;

	t = t.reshape({ 3, 1 });
	torch::Tensor T = torch::eye(4);
	T.index({ slice(0,3),slice(3,4) }) = t;

	torch::Tensor C2W = torch::inverse(Rt);
	torch::Tensor cam_center = C2W.index({ slice(0,3),slice(3,4) });
	cam_center = (cam_center + translate.unsqueeze(1)) * scale;
	C2W.index({ slice(0,3),slice(3,4) }) = cam_center;
	Rt = torch::inverse(C2W);
	Rt = T.matmul(Rt);
	return Rt.to(torch::kFloat32);
}


torch::Tensor getProjectionMatrix(float znear, float zfar, float fovX, float fovY) {
    float tanHalfFovY = tan((fovY / 2));
    float tanHalfFovX = tan((fovX / 2));

    float top = tanHalfFovY * znear;
    float bottom = -top;
    float right = tanHalfFovX * znear;
    float left = -right;

    torch::Tensor P = torch::zeros({ 4, 4 });

    float z_sign = 1.0;
    P.index({ slice(0,1),slice(0,1) }) = torch::tensor({ 2.0 * znear / (right - left) }).unsqueeze(1);
    P.index({ slice(1,2),slice(1,2) }) = torch::tensor({2.0 * znear / (top - bottom)}).unsqueeze(1);
    P.index({ slice(0,1),slice(2,3) }) = torch::tensor({(right + left) / (right - left)}).unsqueeze(1);
    P.index({ slice(1,2),slice(2,3) }) = torch::tensor({(top + bottom) / (top - bottom)}).unsqueeze(1);
    P.index({ slice(3,4),slice(2,3) }) = torch::tensor({z_sign}).unsqueeze(1);
    P.index({ slice(2,3),slice(2,3) }) = torch::tensor({z_sign * zfar / (zfar - znear)}).unsqueeze(1);
    P.index({ slice(2,3),slice(3,4) }) = torch::tensor({-(zfar * znear) / (zfar - znear)}).unsqueeze(1);
    return P;
}


double focal2fov(double focal, double pixels) {
	return 2 * atan(pixels / (2 * focal));
}


void resizeCamera(cameraPose* cp, float factor) {
	cp->pose.index({ torch::indexing::Slice(),torch::indexing::Slice(4) }) /= factor;
}


void resizeCameras(std::vector<cameraPose>& cameraposes, float factor) {
	for (int i = 0;i < cameraposes.size();i++)
		resizeCamera(&cameraposes[i], factor);
}


torch::Tensor my_l1_loss(torch::Tensor network_output, torch::Tensor gt) {
    return torch::abs((network_output - gt)).mean();
}


torch::Tensor gaussian(int window_size, float sigma) {
	// 创建一个一维张量，其大小为window_size
	torch::Tensor gauss = torch::zeros({ window_size },torch::kFloat32);

	// 计算高斯核
	for (int x = 0; x < window_size; ++x) {
		gauss[x] = std::exp(-(x - window_size / 2.0) * (x - window_size / 2.0) / (2 * sigma * sigma));
	}

	// 归一化高斯核
	gauss /= gauss.sum();

	return gauss;
}


torch::Tensor create_window(int window_size, int channel) {
    auto _1D_window = gaussian(window_size, 1.5).unsqueeze(1);
    auto _2D_window = _1D_window.mm(_1D_window.t().to(torch::kFloat32)).unsqueeze(0).unsqueeze(0);
    auto window = _2D_window.expand({ channel, 1, window_size, window_size }).contiguous();
    return window;
}



torch::Tensor _ssim(torch::Tensor img1, torch::Tensor img2, torch::Tensor window, int window_size, int channel) {
    bool size_average = true;
	// 确保输入图像是float类型
    img1 = img1.to(torch::kFloat32);
	img2 = img2.to(torch::kFloat32);

    img1 = img1.permute({2,0,1});
    img2 = img2.permute({ 2,0,1 });

    window = window.to(device);

	// 计算均值
    torch::Tensor mu1 = torch::conv2d(img1, window, {} ,1,window_size / 2, 1, channel);
	torch::Tensor mu2 = torch::conv2d(img2, window, {}, 1, window_size / 2, 1, channel);

	// 计算均值的平方
	torch::Tensor mu1_sq = mu1.pow(2);
	torch::Tensor mu2_sq = mu2.pow(2);
	torch::Tensor mu1_mu2 = mu1 * mu2;

	// 计算方差和协方差
	torch::Tensor sigma1_sq = torch::conv2d(img1*img1, window, {}, 1, window_size / 2, 1, channel) - mu1_sq;
	torch::Tensor sigma2_sq = torch::conv2d(img2*img2, window, {}, 1, window_size / 2, 1, channel) - mu2_sq;
	torch::Tensor sigma12 = torch::conv2d(img1*img2, window, {}, 1, window_size / 2, 1, channel) - mu1_mu2;

	// 常数C1和C2
	double C1 = 0.01 * 0.01;
	double C2 = 0.03 * 0.03;

	// 计算SSIM
	torch::Tensor ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

	// 根据size_average决定返回值
	if (size_average) {
		return ssim_map.mean();
	}
	else {
		return ssim_map.mean(1).mean(1).mean(1);
	}
}


torch::Tensor ssim(torch::Tensor img1, torch::Tensor img2 ) {
    int window_size = 11;
    bool size_average = true;
    int channel = img1.sizes()[2];
    auto window = create_window(window_size, channel);
    return _ssim(img1, img2, window, window_size, channel);
}


torch::Tensor viewMatrix(torch::Tensor z, torch::Tensor up, torch::Tensor pos) {
	torch::Tensor vec2 = z;
	torch::Tensor vec1_avg = up;
	torch::Tensor vec0 = TensorCross(vec1_avg, vec2);//计算两个一维张量的叉积
	vec0 /= torch::norm(vec0, 2);
	torch::Tensor vec1 = TensorCross(vec2, vec0);
	vec1 /= torch::norm(vec1, 2);
	torch::Tensor vm = torch::stack({ vec0, vec1, vec2, pos }, 1).squeeze();
	return vm;//T (3,4)
}



torch::Tensor TensorCross(torch::Tensor a, torch::Tensor b) {
	// 计算叉乘
	torch::Tensor cross_product = torch::stack({
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0]
		}, 0);
	return cross_product;
}


torch::Tensor qvec2rotmat(torch::Tensor& qvec) {

	torch::Tensor rotmat = torch::zeros({ 3, 3 });

	// 直接使用公式计算旋转矩阵的每个元素
	rotmat[0][0] = 1 - 2 * qvec[2] * qvec[2] - 2 * qvec[3] * qvec[3];
	rotmat[0][1] = 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3];
	rotmat[0][2] = 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2];

	rotmat[1][0] = 2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3];
	rotmat[1][1] = 1 - 2 * qvec[1] * qvec[1] - 2 * qvec[3] * qvec[3];
	rotmat[1][2] = 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1];

	rotmat[2][0] = 2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2];
	rotmat[2][1] = 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1];
	rotmat[2][2] = 1 - 2 * qvec[1] * qvec[1] - 2 * qvec[2] * qvec[2];

	return rotmat;
}