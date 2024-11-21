#pragma once
#include"util.h"
#include"GaussianModel.h"
#include"scene.h"
#include"renderer.h"
#define baseDir std::string("C:\\mytinyrenderproj\\MyRenderer\\3dgs\\")//工程目录
#define expName std::string("exps\\Bonsai")//工程目录下的实验目录
bool onlyRender = false;
int main(int argc, char** argv) {
    try {

        std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
        std::cout << "torch::cuda::cudnn_is_available():" << torch::cuda::cudnn_is_available() << std::endl;
        std::cout << "torch::cuda::device_count():" << torch::cuda::device_count() << std::endl;


		int globle_step = 0;
		int epoch = 0;


        //初始化高斯类
        GaussianModel gs(3);

        //初始化场景类
        ModelParameters mp;
        bool ckpt=false;
        if (gs.find_max_tar_file(baseDir, expName, "gs")!="") {
			//加载检查点
            ckpt = true;
			gs.loadCkpt(baseDir, expName, &globle_step, &epoch);
        }
        Scene scene(mp, gs,ckpt);


        //训练初始化
        OptimizationParameters op;
        if(!ckpt)
            gs.training_setup(op);


		Camera viewCamera(scene.Cameras[0]);
        torch::Tensor bg = torch::tensor({ 0,0,0 }, torch::kFloat32).to(device);
        //训练
        while (1) {

            //获取随机相机
            Camera c = scene.getRandomCamera();


            //每过多少帧就更新npy
            if (globle_step % 1 == 0) {
                std::fstream file(baseDir + expName + "\\" + "flag.txt", std::ios::in | std::ios::out);
                file.seekg(0);
                char flag, trans;
                int dx = 0;int dy=0;
                char dxstr,dystr;
                file >> flag >> trans;
                file >> dxstr >> dystr;
                dx = std::atoi(&dxstr)-1;
                dy = std::atoi(&dystr)-1;
                if (flag == '0') {
                    scene.updateViewCamera(trans, &viewCamera,&(scene.Cameras[0]),dx,dy);
                    auto render_view_pkg = render(viewCamera, gs, bg, 1.0);
                    auto color_view = render_view_pkg[0].permute({ 1,2,0 });

                    color_view = color_view.to(torch::kCPU);
                    auto color_contiguous = color_view.contiguous().clone();
                    float* data = color_contiguous.data_ptr<float>();
                    std::vector<float>datavec(data, data + viewCamera.image_width * viewCamera.image_height * 3);
                    std::string posergbPath = baseDir + expName + "/" + "posergb.npy";
                    std::cout << "正在保存" << std::endl;
                    cnpy::npy_save(posergbPath, datavec);
                    file.seekp(0);
                    file.seekg(0);
                    file << std::to_string(1);
                    file.flush();
                }
                file.close();
      
            }

            if (!onlyRender) {

                globle_step++;

                gs.update_lr_xyz(globle_step);

                //渲染（前向传播）
                auto render_pkg = render(c, gs, bg, 1.0);
                auto color = render_pkg[0].permute({ 1,2,0 });
                auto radii = render_pkg[1];
                auto viewspace_point_tensor = render_pkg[2];//这是一个只有梯度的张量
                auto visibility_index = torch::nonzero(torch::where(radii > 0, torch::ones_like(radii), torch::zeros_like(radii))).squeeze().to(device);
      
                //计算loss并反向传播
                auto gt_image = c.original_image;
                auto Ll1 = my_l1_loss(color, gt_image);
                auto loss = (1.0 - op.lambda_dssim) * Ll1 + op.lambda_dssim * (1.0 - ssim(color, gt_image));
                print(std::to_string(globle_step) + "   ", loss);
                loss.backward();

                {
                    at::NoGradGuard nograd;
                    //更新参数
                    gs.optimizer->step();
                    gs.optimizer->zero_grad();

                    //Densification
                    if (globle_step < op.densify_until_iter) {
                        gs.max_radii2D.index_put_({ visibility_index }, torch::max(gs.max_radii2D.index_select({ 0 }, visibility_index), radii.index_select({ 0 }, visibility_index)));//每个高斯的历史最大radii
                        gs.add_densification_stats(viewspace_point_tensor, visibility_index);//累加每个高斯的means2d梯度的2L范数，xyz_gradient_accum存储累计梯度，denom存储累计数量，这样就可以求历史平均梯度
                        if (globle_step >500 && globle_step % 100 == 0) {
                            int preNum = gs.xyz.sizes()[0];
                            //对高斯的密度进行优化
                            int  size_threshold = (globle_step > 3000) ? 20 : 0;
                            gs.densify_and_prune(op.densify_grad_threshold, 0.005, scene.camera_extent, size_threshold);

                            std::cout << "增加了" << gs.xyz.sizes()[0] - preNum << "个高斯" << std::endl;
                            std::cout << "现在有" << gs.xyz.sizes()[0] << "个高斯" << std::endl;
                        }
                        if (globle_step % 3000 == 0) {
                            gs.reset_opacity();
                        }
                    }
                }

                if (globle_step > 5000 && globle_step % 1000 == 0)
                    gs.saveCkpt(baseDir, expName, globle_step, globle_step / scene.Cameras.size());

                if (globle_step % 1000 == 0) {
                    if (gs.active_sh_degree < gs.max_sh_degree)
                        gs.active_sh_degree += 1;
                }

            }
        }
        return 0;
    }
    catch (const c10::Error& e) {
        std::cout << e.msg() << std::endl;
    }
}


