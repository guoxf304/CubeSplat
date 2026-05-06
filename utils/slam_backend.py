# 在导入任何模块之前设置环境变量，避免Tkinter问题
import os
os.environ['MPLBACKEND'] = 'Agg'  # matplotlib使用非交互式后端
os.environ['PIL_USE_TKINTER'] = '0'  # 禁用PIL的Tkinter支持

# 在导入任何模块之前设置stderr过滤器，捕获所有Tkinter错误
import sys
_original_stderr = sys.stderr
class _FilteredStderr:
    def __init__(self, original):
        self.original = original
        self._buffer = ''
    def write(self, message):
        # 累积多行消息
        self._buffer += message
        # 检查是否包含Tkinter相关错误
        msg_lower = self._buffer.lower()
        if ('exception ignored' in msg_lower or 'exception ignored in:' in msg_lower) and (
            'tkinter' in msg_lower or 
            'main thread is not in main loop' in msg_lower or
            'image.__del__' in msg_lower or
            'variable.__del__' in msg_lower or
            'tkinter/__init__.py' in self._buffer
        ):
            # 清空缓冲区，不输出
            self._buffer = ''
            return
        # 如果缓冲区有内容且不是Tkinter错误，输出并清空
        if self._buffer:
            self.original.write(self._buffer)
            self._buffer = ''
    def flush(self):
        if self._buffer and not any(keyword in self._buffer.lower() for keyword in ['tkinter', 'main thread is not in main loop', 'image.__del__', 'variable.__del__']):
            self.original.write(self._buffer)
        self._buffer = ''
        self.original.flush()
    def __getattr__(self, name):
        return getattr(self.original, name)
sys.stderr = _FilteredStderr(_original_stderr)

import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping
import numpy as np
# from utils.cubemap import tensor_plot

class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config # 存储配置信息 (`config`)，包括训练参数、相机参数等
        self.gaussians = None # 存储 3D 高斯点云模型
        self.pipeline_params = None # 存储 数据处理的流水线参数（如 渲染管道）
        self.opt_params = None # 存储 优化参数（用于 Adam 或 SGD 进行 SLAM 计算）
        self.background = None # 存储 背景颜色
        self.frontend_queue = None # 前端数据队列，存储从 `FrontEnd` 发送到 `BackEnd` 的数据
        self.backend_queue = None # 后端数据队列，存储 `BackEnd` 传递给 `FrontEnd` 的指令（如暂停、优化等）
        self.live_mode = False # 是否是实时模式，False离线模式，预加载数据集，True实时在线模式，如传感器

        self.pause = False # 是否暂停
        self.device = "cuda" # 使用 cuda
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"] # 是否为单目
        self.iteration_count = 0 # SLAM运行的总迭代次数
        self.last_sent = 0 #  最后一次传输关键帧的时间戳
        self.occ_aware_visibility = {} # 存储 物体遮挡信息
        self.viewpoints = {} # 存储 所有相机视角信息
        self.current_window = [] # 当前优化窗口的关键帧索引
        self.initialized = not self.monocular # 如果是单目，需要初始化尺度恢复之后才能运行，否则，立即执行初始化
        self.keyframe_optimizers = None # 关键帧优化器
        self.gaussian_num = []
        

    # 从配置中读取超参数，设置高斯模型的训练过程中的各类参数，包括高斯模型更新、映射迭代等
    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"] # bool，是否保存结果

        # 高斯点云初始化参数
        self.init_itr_num = self.config["Training"]["init_itr_num"] # 高斯点云初始化迭代次数
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"] # 是否在初始化阶段更新高斯点云（True/False）
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"] # 初始化时是否重置高斯点云
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"] # 高斯点云初始化的阈值（决定哪些点可以参与初始化）
        self.init_gaussian_extent = ( # 高斯点云初始化的空间范围
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        # 高斯点云更新参数
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"] # 地图优化的迭代次数
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"] # 每 N 帧更新一次高斯点云
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"] # 偏移量，用于调整高斯点云的更新节奏
        self.gaussian_th = self.config["Training"]["gaussian_th"] # 高斯点云更新的过滤阈值
        self.gaussian_extent = ( # 高斯点云的范围
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"] # 否在 SLAM 运行时重置高斯点云
        self.size_threshold = self.config["Training"]["size_threshold"] # 高斯点云的最小尺寸

        self.window_size = self.config["Training"]["window_size"] # 滑动窗口优化的大小
        self.single_thread = ( # 单线程 SLAM，False表示多线程
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )

    def _should_apply_quantization(self):
        """判断是否应该进行量化"""
        if not self.gaussians.quantization_enabled:
            return False
            
        # 检查是否已经初始化完成
        if not self.initialized:
            return False
            
        return True

    # 将新关键帧添加到高斯模型中，并根据深度图进行初始化
    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map_dict=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap_dict=depth_map_dict
        )
        # 注意：量化移动到关键帧处理完成之后（run() 的 keyframe 分支末尾）

    # 重置系统
    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        if self.gaussians is not None:
            self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()






    # 初始化地图
    def initialize_map(self, cur_frame_idx, viewpoint):
        start_time = time.time()
        self.init_itr_num = 550
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1 # 每运行一次迭代次数加一

            total_loss = 0
            render_pkgs = {}  # 保存每个面渲染结果

            for face_key in viewpoint.Cubemap_image:
                render_pkg = render(  # 渲染当前帧（后端使用未量化）
                    viewpoint, self.gaussians, self.pipeline_params, self.background, face_key=face_key, use_quantized=False
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],  # 渲染后的RGB图像
                    render_pkg["viewspace_points"],  # 3D高斯点云投影到viewspace
                    render_pkg["visibility_filter"],  # 当前帧可见Gaussian的Mask
                    render_pkg["radii"],  # 投影后的Gaussian半径
                    render_pkg["depth"],  # 渲染出的深度图
                    render_pkg["opacity"],  # 点云Alpha透明图
                    render_pkg["n_touched"],  # 被ViewPoint观测到的Gaussians计数
                )
                # 每个面单独计算 loss
                # 如果有预测的深度，使用 get_loss_mapping_rgbd，否则使用原来的逻辑
                if hasattr(viewpoint, 'depth') and viewpoint.depth is not None:
                    if isinstance(viewpoint.depth, dict) and face_key in viewpoint.depth:
                        # 有预测深度，使用 get_loss_mapping_rgbd
                        from utils.slam_utils import get_loss_mapping_rgbd
                        loss_init = get_loss_mapping_rgbd(
                            self.config, image, depth, viewpoint, initialization=True, face_key=face_key
                        )
                    else:
                        # 没有该面的深度，使用原来的逻辑
                        loss_init = get_loss_mapping(
                            self.config, image, depth, viewpoint, opacity, initialization=True, face_key=face_key
                        )
                else:
                    # 没有预测深度，使用原来的逻辑
                    loss_init = get_loss_mapping( # 计算loss
                        self.config, image, depth, viewpoint, opacity, initialization=True, face_key = face_key
                    )
                total_loss += loss_init  # 叠加四面 loss

                # 暂存每面数据，用于后面 visibility 统计
                render_pkgs[face_key] = render_pkg

            total_loss.backward()  # 总loss反向传播


            with torch.no_grad(): # 更新3D GS
                for face_key in viewpoint.Cubemap_image:
                    render_info = render_pkgs[face_key]
                    viewspace_point_tensor = render_info["viewspace_points"]
                    visibility_filter = render_info["visibility_filter"]
                    radii = render_info["radii"]

                    # 更新 max_radii2D
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    # 添加 Density Statistics
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor, visibility_filter
                    )

                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        # 设置可见性
        total_n_touched = None
        for face_key in viewpoint.Cubemap_image:
            n_touched = render_pkgs[face_key]["n_touched"]
            if total_n_touched is None:
                total_n_touched = n_touched.clone()
            else:
                total_n_touched += n_touched  # 逐元素累加被观测到的次数
        self.occ_aware_visibility[cur_frame_idx] = total_n_touched.long()
        Log("Initialized map") # SLAM初始化完成
        loop_time = time.time() - start_time
        print(f"初始化耗时: {loop_time:.2f} 秒")
        return render_pkgs

    # 地图优化
    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window] # 当前窗口的帧
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"] # 控制帧的数量

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint) # 不在当前窗口的帧

        for _ in range(iters): # 优化
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            total_n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                total_loss_mapping = 0
                total_n_touched = None
                render_pkgs = {}
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)

                for face_key in viewpoint.Cubemap_image:
                    render_pkg = render(
                        viewpoint, self.gaussians, self.pipeline_params, self.background, face_key=face_key, use_quantized=False
                    )

                    (
                        image,
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                        depth,
                        opacity,
                        n_touched,
                    ) = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                        render_pkg["depth"],
                        render_pkg["opacity"],
                        render_pkg["n_touched"],
                    )

                    loss_mapping += get_loss_mapping(
                        self.config, image, depth, viewpoint, opacity, face_key=face_key
                    )
                    viewspace_point_tensor_acm.append(viewspace_point_tensor)
                    visibility_filter_acm.append(visibility_filter)
                    radii_acm.append(radii)
                    if total_n_touched is None:
                        total_n_touched = n_touched.clone()
                    else:
                        total_n_touched += n_touched

                total_n_touched_acm.append(total_n_touched)



            # 渲染随机视角
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                for face_key in viewpoint.Cubemap_image:
                    render_pkg = render(
                        viewpoint, self.gaussians, self.pipeline_params, self.background, face_key=face_key, use_quantized=False
                    )
                    (
                        image,
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                        depth,
                        opacity,
                        n_touched,
                    ) = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                        render_pkg["depth"],
                        render_pkg["opacity"],
                        render_pkg["n_touched"],
                    )
                    loss_mapping += get_loss_mapping(
                        self.config, image, depth, viewpoint, opacity, face_key=face_key
                    )
                    viewspace_point_tensor_acm.append(viewspace_point_tensor)
                    visibility_filter_acm.append(visibility_filter)
                    radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False

            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    total_n_touched = total_n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (total_n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)

                # 在每次迭代后，仅更新量化聚类中心（assign=False），确保渲染使用最新中心
                # 只有在量化中心已存在时才进行更新（即量化初始化完成后）
                # if self.gaussians.kmeans_quantizers and any(center is not None for center in self.gaussians.quantization_centers.values()):
                #     try:
                #         self.gaussians.apply_quantization(param_types=None, assign_flag = False)
                #     except Exception as e:
                #         print(f"量化中心更新失败: {e}")
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        return gaussian_split

    # 颜色优化（适配cubemap模式，参考map函数，一次性优化一帧的各个面）
    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            # 随机选择一个视角
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            
            # 检查该视角是否有可用的面
            if not hasattr(viewpoint_cam, 'Cubemap_image') or len(viewpoint_cam.Cubemap_image) == 0:
                continue
            
            # 初始化累加变量
            total_loss = 0
            visibility_filter_acm = []
            radii_acm = []
            
            # 遍历该视角的所有面，分别渲染并计算损失
            for face_key in viewpoint_cam.Cubemap_image:
                # 渲染该面
                render_pkg = render(
                    viewpoint_cam, self.gaussians, self.pipeline_params, self.background, face_key=face_key
                )
                image, visibility_filter, radii = (
                    render_pkg["render"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )

                # 获取该面的真实图像
                gt_image = viewpoint_cam.Cubemap_image[face_key].cuda()
                Ll1 = l1_loss(image, gt_image)
                face_loss = (1.0 - self.opt_params.lambda_dssim) * (
                    Ll1
                ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
                
                # 累加损失
                total_loss += face_loss
                
                # 保存visibility_filter和radii用于后续更新
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
            
            # 统一进行反向传播
            total_loss.backward()
            
            # 更新参数
            with torch.no_grad():
                # 更新max_radii2D（参考map函数的实现）
                for idx in range(len(visibility_filter_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                )
                
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")


    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)
    
    def push_gaussians_to_frontend(self):
        """专门用于推送gaussians到frontend，用于子图保存"""
        self.push_to_frontend("sync_backend")

    # 后端主循环
    def run(self):
        # 在子进程启动时设置环境变量和异常处理
        import warnings
        import sys
        import traceback
        
        os.environ['MPLBACKEND'] = 'Agg'
        os.environ['PIL_USE_TKINTER'] = '0'
        
        warnings.filterwarnings('ignore', message='.*tkinter.*', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='.*main thread is not in main loop.*', category=RuntimeWarning)
        
        # 设置异常钩子
        original_excepthook = sys.excepthook
        def custom_excepthook(exc_type, exc_value, exc_traceback):
            if exc_type == RuntimeError:
                error_msg = str(exc_value)
                if 'main thread is not in main loop' in error_msg:
                    return
                if exc_traceback is not None:
                    tb_str = ''.join(traceback.format_tb(exc_traceback))
                    if 'tkinter' in tb_str.lower():
                        return
            original_excepthook(exc_type, exc_value, exc_traceback)
        sys.excepthook = custom_excepthook
        
        # stderr过滤器已在文件开头设置，这里不需要重复设置
        
        while True:
            # 监听backend_queue，若空
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue
                if self.single_thread:
                    time.sleep(0.01)
                    continue

                # 允许非关键帧背景优化，维持前端及时获得最新gaussian
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            # 若backend_queue队列非空
            else:
                # 获取前端的指令，然后进行相应的处理

                data = self.backend_queue.get()
                if data[0] == "stop": # 停止后端
                    break
                elif data[0] == "pause": # 暂停
                    self.pause = True
                elif data[0] == "unpause": # 恢复
                    self.pause = False
                elif data[0] == "push_gaussians": # 推送gaussians到frontend（用于子图保存）
                    self.push_gaussians_to_frontend()
                elif data[0] == "color_refinement": # 颜色优化
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init": # 处理初始化
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset() # 重置系统

                    self.viewpoints[cur_frame_idx] = viewpoint
                    # 将预测深度设置到 viewpoint.depth，以便在 initialize_map 中使用
                    if depth_map is not None and len(depth_map) > 0:
                        # 将 numpy 数组转换为 torch.Tensor（如果需要）
                        depth_dict = {}
                        for key, depth_np in depth_map.items():
                            if isinstance(depth_np, np.ndarray):
                                depth_dict[key] = torch.from_numpy(depth_np).float().to(viewpoint.device)
                            else:
                                depth_dict[key] = depth_np
                        viewpoint.depth = depth_dict
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map_dict=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                # 受到`keyframe`指令，
                elif data[0] == "keyframe":
                    start_time = time.time()
                    num1 = self.gaussians.get_xyz.shape[0]
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map_dict = data[4]

                    # 关键帧优化
                    self.viewpoints[cur_frame_idx] = viewpoint # 存储关键帧
                    self.current_window = current_window # 更新current_windows（关键帧窗口）
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map_dict=depth_map_dict) # 添加关键帧到SLAM系统
                    num2 = self.gaussians.get_xyz.shape[0]

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"] # 优化窗口大小（pose_window）
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    # 如果SLAM还未初始化
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"] # 关键帧数达到window_size，执行BA优化，提高位姿精度
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            # 迭代次数：实时模式50次优化，非实时模式，300次优化
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    # 设置优化参数
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta], # 优化相机旋转
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a], # 曝光
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b], # 曝光
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    # 使用Adam优化器进行关键帧优化
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    # 地图优化
                    self.map(self.current_window, iters=iter_per_kf)
                    num3 = self.gaussians.get_xyz.shape[0]
                    self.map(self.current_window, prune=True)
                    num4 = self.gaussians.get_xyz.shape[0]
                    # 在推送给前端之前进行量化，这样前端能拿到最新的量化结果
                    if self._should_apply_quantization():
                        try:
                            print(f"开始量化第{cur_frame_idx}帧")
                            self.gaussians.apply_quantization(assign_flag = True)
                            print(f"量化第{cur_frame_idx}帧完成")
                            self.gaussians.verify_quantization_effectiveness()
                        except Exception as e:
                            print(f"量化失败: {e}")

                    self.push_to_frontend("keyframe") # 向前端推送`keyframe`优化结果

                    loop_time = time.time() - start_time
                    self.gaussian_num.append(self.gaussians.get_xyz.shape[0])
                    print(
                        f'处理关键帧所花时间: {loop_time:.2f}秒 |'  # 保留2位小数
                        f'num1: {num1} | '
                        f'num2: {num2} | '
                        f'num3: {num3} | '
                        f'num4: {num4} | '
                    )
                else:
                    raise Exception("Unprocessed data", data)
        # 清理消息队列
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
