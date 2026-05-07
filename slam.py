# 在所有导入之前设置环境变量，避免Tkinter/PIL在多进程环境中的问题
# 必须在导入任何可能使用Tkinter的库之前设置
import os
os.environ['MPLBACKEND'] = 'Agg'  # matplotlib使用非交互式后端
os.environ['PIL_USE_TKINTER'] = '0'  # 禁用PIL的Tkinter支持

import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb


from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd
from utils.cubemap import  PALToCubemapConverter, EquirectangularToCubemapConverter, ERPToCube
from utils.submap_manager import SubmapManager

class SLAM:
    def __init__(self, config, save_dir=None):
        # 创建两个CUDA对象，用于测量时间
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.isCube = True
        self.faces = ['front', 'back', 'left', 'right']

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense" # 是否是实时 模式
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular" # 单目传感器
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"] # 是否使用球谐函数增强光照建模能力，false
        # 读取坐标系转换配置（360VO需要转换，PAL不需要）
        self.coordinate_transform = self.config["Dataset"].get("coordinate_transform", False)
        self.use_gui = self.config["Results"]["use_gui"] # 是否使用gui可视化界面，默认true
        # 如果是实时模式，则强制使用GUI
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]
        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0
        # 如果 self.use_spherical_harmonics 为 True，那么 model_params.sh_degree 设为 3，否则为0，表示球谐函数展开的阶数
        # 初始化高斯模型
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.faces = self.faces # 赋值给 gaussians
        self.gaussians.init_lr(6.0) # 初始化学习
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )
        self.dataset.faces = self.faces

        # 设置高斯模型的训练
        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        # 背景颜色为黑色
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 初始化前端frontend和后端backend
        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        # 使用多进程队列进行前后端通信
        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular


        start_time = time.time()
        if config["Dataset"]["type"] == "LF-VIO" or config["Dataset"]["type"] == "LF-VISLAM":
            self.Cube = PALToCubemapConverter(self.dataset)
        elif config["Dataset"]["type"] == "ERP":
            self.Cube = ERPToCube(self.dataset)
        loop_time = time.time() - start_time
        # print(f"Time of Cube: {loop_time:.2f} s")

        self.frontend = FrontEnd(self.config) # 创立前端
        self.backend = BackEnd(self.config) # 创立后端

        # 关联 Frontend 组件的数据、队列和超参数

        self.frontend.Cube = self.Cube  # PAL 转化模型
        self.frontend.faces = self.faces
        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        # 关联 Backend 组件的数据、队列和超参数
        self.backend.gaussians = self.gaussians
        self.backend.faces = self.faces
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        # 初始化子图管理器（在主进程中）
        self.submap_manager = SubmapManager(self.config, save_dir=save_dir)
        # 关联前后端引用（用于访问数据）
        self.submap_manager.backend = self.backend
        self.submap_manager.frontend = self.frontend
        # 将子图管理器传递给前端
        self.frontend.submap_manager = self.submap_manager

        # 创建 GUI 组件 ParamsGUI，用于可视化参数和 3D 高斯点云
        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
            Cube = self.Cube,
        )


        if self.use_gui:
            # 如果使用GUI，启动GUI进程
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(2)
        # 后段进程
        backend_process = mp.Process(target=self.backend.run)


        backend_process.start()
        # 前段进程
        self.frontend.run() # 启动 Frontend 处理输入数据流
        backend_queue.put(["pause"]) # 通知 Backend 暂停

        end.record() # 记录结束时间
        torch.cuda.synchronize()

        # empty the frontend queue
        # 计算帧率（FPS），用于评估 SLAM 的性能
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        total_time = start.elapsed_time(end) * 0.001  # 转换为秒
        Log("Total time", total_time, tag="Eval")
        Log("Total FPS", FPS, tag="Eval")

        # 初始化ATE变量
        ATE_result = None
        ATE = None

        # 先计算ATE（用于results.txt），无论eval_rendering是否为True
        try:
            ATE_result = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
                coordinate_transform=self.coordinate_transform if self.eval_rendering else False,
            )
            ATE = ATE_result["ate_kf"] if ATE_result else None
        except Exception as e:
            Log(f"Failed to calculate ATE: {e}", tag="Eval")
            ATE_result = None
            ATE = None

        # 立即生成results.txt文件（在log FPS之后）
        if self.save_dir is not None:
            results_file = os.path.join(self.save_dir, "results.txt")
            with open(results_file, "w", encoding="utf-8") as f:
                f.write("=" * 50 + "\n")
                f.write("SLAM Performance Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Frames: {N_frames}\n")
                f.write(f"Total Time: {total_time:.4f} seconds\n")
                f.write(f"Frame Rate (FPS): {FPS:.4f}\n\n")
                
                # 关键帧信息
                if ATE_result is not None:
                    kf_count = ATE_result.get("kf_count", 0)
                    kf_ids = ATE_result.get("kf_ids", [])
                    f.write("-" * 50 + "\n")
                    f.write("Keyframe Information\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Keyframe Count: {kf_count}\n")
                    f.write(f"Keyframe IDs: {', '.join(map(str, kf_ids))}\n\n")
                    
                    # 关键帧位姿评估
                    f.write("-" * 50 + "\n")
                    f.write("Pose Evaluation - Keyframes\n")
                    f.write("-" * 50 + "\n")
                    ate_kf = ATE_result.get("ate_kf")
                    if ate_kf is not None:
                        f.write(f"Absolute Trajectory Error (ATE): {ate_kf:.6f} m\n")
                    else:
                        f.write(f"Absolute Trajectory Error (ATE): N/A\n")
                    f.write("\n")
                    
                    # 所有帧位姿评估
                    f.write("-" * 50 + "\n")
                    f.write("Pose Evaluation - All Frames\n")
                    f.write("-" * 50 + "\n")
                    ate_all = ATE_result.get("ate_all")
                    if ate_all is not None:
                        f.write(f"Absolute Trajectory Error (ATE): {ate_all:.6f} m\n")
                    else:
                        f.write(f"Absolute Trajectory Error (ATE): N/A\n")
                else:
                    f.write("-" * 50 + "\n")
                    f.write("Pose Evaluation\n")
                    f.write("-" * 50 + "\n")
                    if ATE is not None:
                        f.write(f"Absolute Trajectory Error (ATE): {ATE:.6f} m\n")
                    else:
                        f.write(f"Absolute Trajectory Error (ATE): N/A\n")
                
                f.write("\n" + "=" * 50 + "\n")
            Log(f"Results saved to {results_file}", tag="Eval")

        # 如果eval_rendering为True，进行渲染评估
        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices
            # ATE已经在上面计算过了，这里直接使用
            # 评估渲染质量（cubemap模型，每个面分别计算指标）
            rendering_result_before = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
                cube=self.Cube
            )
            # 从结果中提取所有面的信息
            faces = []
            for key in rendering_result_before.keys():
                if key.endswith("_psnr") and key != "mean_psnr":
                    faces.append(key.replace("_psnr", ""))
            faces = sorted(faces)  # 排序以保证一致性
            
            # 创建并记录度量表（包含每个面的指标）
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            for face in faces:
                columns.extend([f"{face}_psnr", f"{face}_ssim", f"{face}_lpips"])
            metrics_table = wandb.Table(columns=columns)
            
            # 添加优化前的数据
            row_data = [
                "Before",
                rendering_result_before["mean_psnr"],
                rendering_result_before["mean_ssim"],
                rendering_result_before["mean_lpips"],
                ATE,
                FPS,
            ]
            for face in faces:
                row_data.extend([
                    rendering_result_before.get(f"{face}_psnr", 0.0),
                    rendering_result_before.get(f"{face}_ssim", 0.0),
                    rendering_result_before.get(f"{face}_lpips", 0.0),
                ])
            metrics_table.add_data(*row_data)

            # re-used the frontend queue to retrive the gaussians from the backend.
            # 清空前端队列并请求后端进行颜色优化
            while not frontend_queue.empty():
                frontend_queue.get()
            backend_queue.put(["color_refinement"])
            while True:
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()
                if data[0] == "sync_backend" and frontend_queue.empty():
                    gaussians = data[1]
                    self.gaussians = gaussians
                    break

            # 评估优化后的渲染结果（cubemap模型，每个面分别计算指标）
            extra_fov_deg = self.config.get("Rendering", {}).get("extra_fov_deg", 95)
            rendering_result_after = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="after_opt",
                cube=self.Cube,
                extra_fov_deg=extra_fov_deg,
            )
            
            # 添加优化后的数据
            row_data = [
                "After",
                rendering_result_after["mean_psnr"],
                rendering_result_after["mean_ssim"],
                rendering_result_after["mean_lpips"],
                ATE,
                FPS,
            ]
            for face in faces:
                row_data.extend([
                    rendering_result_after.get(f"{face}_psnr", 0.0),
                    rendering_result_after.get(f"{face}_ssim", 0.0),
                    rendering_result_after.get(f"{face}_lpips", 0.0),
                ])
            metrics_table.add_data(*row_data)
            wandb.log({"Metrics": metrics_table})
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)
            
            # 将渲染评估结果追加到results.txt文件
            if self.save_dir is not None:
                results_file = os.path.join(self.save_dir, "results.txt")
                with open(results_file, "a", encoding="utf-8") as f:
                    f.write("\n")
                    f.write("=" * 50 + "\n")
                    f.write("Rendering Quality Evaluation\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # 优化前的评估结果
                    f.write("-" * 50 + "\n")
                    f.write("Before Color Refinement\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Overall PSNR: {rendering_result_before['mean_psnr']:.4f} dB\n")
                    f.write(f"Overall SSIM: {rendering_result_before['mean_ssim']:.4f}\n")
                    f.write(f"Overall LPIPS: {rendering_result_before['mean_lpips']:.4f}\n\n")
                    
                    # 每个面的详细指标（优化前）
                    for face in faces:
                        face_psnr = rendering_result_before.get(f"{face}_psnr", 0.0)
                        face_ssim = rendering_result_before.get(f"{face}_ssim", 0.0)
                        face_lpips = rendering_result_before.get(f"{face}_lpips", 0.0)
                        f.write(f"  {face.capitalize()} Face:\n")
                        f.write(f"    PSNR: {face_psnr:.4f} dB\n")
                        f.write(f"    SSIM: {face_ssim:.4f}\n")
                        f.write(f"    LPIPS: {face_lpips:.4f}\n\n")
                    
                    # 优化后的评估结果
                    f.write("-" * 50 + "\n")
                    f.write("After Color Refinement\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Overall PSNR: {rendering_result_after['mean_psnr']:.4f} dB\n")
                    f.write(f"Overall SSIM: {rendering_result_after['mean_ssim']:.4f}\n")
                    f.write(f"Overall LPIPS: {rendering_result_after['mean_lpips']:.4f}\n\n")
                    
                    # 每个面的详细指标（优化后）
                    for face in faces:
                        face_psnr = rendering_result_after.get(f"{face}_psnr", 0.0)
                        face_ssim = rendering_result_after.get(f"{face}_ssim", 0.0)
                        face_lpips = rendering_result_after.get(f"{face}_lpips", 0.0)
                        f.write(f"  {face.capitalize()} Face:\n")
                        f.write(f"    PSNR: {face_psnr:.4f} dB\n")
                        f.write(f"    SSIM: {face_ssim:.4f}\n")
                        f.write(f"    LPIPS: {face_lpips:.4f}\n\n")
                    
                    f.write("=" * 50 + "\n")
                Log(f"Rendering evaluation results appended to {results_file}", tag="Eval")

        # 视频渲染现在在前端tracking过程中实时完成（见 FrontEnd.tracking）

        # 子图管理：保存最后一个子图并拼接所有子图
        if hasattr(self, 'submap_manager') and self.submap_manager is not None:
            # 先检查所有未保存的子图，尝试保存
            # 注意：在SLAM结束时，gaussians可能已经被reset，所以可能无法保存
            # 但至少尝试保存当前活跃的子图
            if self.submap_manager.current_submap_id is not None:
                submap = self.submap_manager.get_current_submap()
                if submap is not None and not submap.is_saved and submap.keyframes:
                    Log(f"Attempting to save final submap {submap.submap_id} at end of SLAM", tag="Submap")
                    self.submap_manager._finalize_current_submap()
                    if self.gaussians is not None:
                        ckpt_path = self.submap_manager.save_dict_to_ckpt(
                            submap.submap_id,
                            self.gaussians,
                            self.frontend.cameras
                        )
                        if ckpt_path:
                            Log(f"Saved final submap {submap.submap_id} at end of SLAM", tag="Submap")
                        else:
                            Log(f"Warning: Failed to save final submap {submap.submap_id} (no gaussians found for keyframes)", tag="Submap")
                    else:
                        Log(f"Warning: Cannot save final submap {submap.submap_id}, gaussians is None", tag="Submap")
            
            # 等待文件写入完成
            time.sleep(0.3)  # 等待300ms确保所有文件写入完成
            
            # 拼接所有子图并保存为PLY文件
            Log("Starting to merge all submaps...", tag="Submap")
            merged_ply_path = self.submap_manager.merge_submaps_to_ply()
            if merged_ply_path:
                Log(f"Merged submaps saved to: {merged_ply_path}", tag="Submap")
            else:
                Log("Failed to merge submaps", tag="Submap")
        
        # 终止backend和GUI进程
        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:]) # fr1_desk.yaml

    # 环境变量已在文件开头设置，这里直接设置多进程启动方法
    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config) # 获得了所有配置
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    slam = SLAM(config, save_dir=save_dir)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
