# 在导入matplotlib之前设置环境变量，避免Tkinter问题
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

import time
import csv
import numpy as np
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import cv2

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gaussian_splatting.utils.image_utils import psnr
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians, save_video_frame_realtime, save_face_render_realtime
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
from utils.depth_replacement import pointmap_replacement
from depth_em.depth_estimator import estimate_depth
# from rotated import plot_2tensor
# from utils.cubemap import tensor_plot
# from utils.debug_function import show_all_render

def save_init_confidence_visualization(viewpoint, cur_frame_idx, save_dir, confidence_threshold=0.4):
    """
    保存初始化时置信度mask下的原始图像可视化
    
    Args:
        viewpoint: 相机视角对象
        cur_frame_idx: 当前帧索引
        save_dir: 保存目录
        confidence_threshold: 置信度阈值（默认0.4，即40%）
    """
    try:
        if not hasattr(viewpoint, 'confidence') or viewpoint.confidence is None:
            return
        
        init_dir = os.path.join(save_dir, "init_confidence_visualization")
        os.makedirs(init_dir, exist_ok=True)
        
        frame_dir = os.path.join(init_dir, f"frame_{cur_frame_idx:06d}")
        os.makedirs(frame_dir, exist_ok=True)
        
        for face_key in viewpoint.Cubemap_image.keys():
            if face_key not in viewpoint.confidence or viewpoint.confidence[face_key] is None:
                continue
            
            # 获取RGB图像
            rgb_img = viewpoint.Cubemap_image[face_key]  # (C, H, W)
            rgb_np = rgb_img.detach().cpu().permute(1, 2, 0).numpy()  # (H, W, C)
            rgb_np = np.clip(rgb_np, 0.0, 1.0)
            
            # 获取深度图
            if viewpoint.depth is not None and face_key in viewpoint.depth:
                depth_tensor = viewpoint.depth[face_key]
                if depth_tensor.dim() == 2:
                    depth_np = depth_tensor.detach().cpu().numpy()
                else:
                    depth_np = depth_tensor.squeeze(0).detach().cpu().numpy()
            else:
                depth_np = None
            
            # 获取置信度图
            conf_tensor = viewpoint.confidence[face_key]
            if conf_tensor.dim() > 2:
                conf_np = conf_tensor.squeeze(0).detach().cpu().numpy()
            else:
                conf_np = conf_tensor.detach().cpu().numpy()
            
            # 创建置信度mask
            high_conf_mask = conf_np > confidence_threshold
            
            # 归一化深度图用于可视化
            if depth_np is not None:
                depth_vis = depth_np.copy()
                depth_vis[depth_vis <= 0] = 0
                if depth_vis.max() > 0:
                    depth_vis = depth_vis / depth_vis.max()
                depth_vis_colored = plt.cm.viridis(depth_vis)[:, :, :3]  # 使用viridis colormap
            else:
                depth_vis_colored = np.zeros_like(rgb_np)
            
            # 创建过滤后的深度图（只保留高置信度区域）
            if depth_np is not None:
                filtered_depth = depth_np.copy()
                filtered_depth[~high_conf_mask] = 0
                filtered_depth_vis = filtered_depth.copy()
                filtered_depth_vis[filtered_depth_vis <= 0] = 0
                if filtered_depth_vis.max() > 0:
                    filtered_depth_vis = filtered_depth_vis / filtered_depth_vis.max()
                filtered_depth_vis_colored = plt.cm.viridis(filtered_depth_vis)[:, :, :3]
            else:
                filtered_depth_vis_colored = np.zeros_like(rgb_np)
            
            # 创建RGB图像叠加置信度mask的可视化
            rgb_with_mask = rgb_np.copy()
            # 将低置信度区域标记为红色半透明
            low_conf_mask = ~high_conf_mask
            rgb_with_mask[low_conf_mask] = rgb_with_mask[low_conf_mask] * 0.5 + np.array([1.0, 0.0, 0.0]) * 0.5
            
            # 创建组合可视化图像
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 第一行：原始RGB、深度图、置信度图
            axes[0, 0].imshow(rgb_np)
            axes[0, 0].set_title(f'Original RGB - {face_key}', fontsize=12)
            axes[0, 0].axis('off')
            
            if depth_np is not None:
                axes[0, 1].imshow(depth_vis_colored)
                axes[0, 1].set_title(f'Original Depth - {face_key}', fontsize=12)
                axes[0, 1].axis('off')
            else:
                axes[0, 1].text(0.5, 0.5, 'No Depth', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].axis('off')
            
            conf_vis = axes[0, 2].imshow(conf_np, cmap='hot', vmin=0, vmax=1)
            axes[0, 2].set_title(f'Confidence Map - {face_key}', fontsize=12)
            axes[0, 2].axis('off')
            plt.colorbar(conf_vis, ax=axes[0, 2], fraction=0.046, pad=0.04)
            
            # 第二行：RGB+Mask、过滤后的深度、统计信息
            axes[1, 0].imshow(rgb_with_mask)
            axes[1, 0].set_title(f'RGB with Confidence Mask\n(Red: <{confidence_threshold*100:.0f}%)', fontsize=12)
            axes[1, 0].axis('off')
            
            if depth_np is not None:
                axes[1, 1].imshow(filtered_depth_vis_colored)
                axes[1, 1].set_title(f'Filtered Depth (conf > {confidence_threshold*100:.0f}%)', fontsize=12)
                axes[1, 1].axis('off')
            else:
                axes[1, 1].text(0.5, 0.5, 'No Depth', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].axis('off')
            
            # 统计信息
            total_pixels = high_conf_mask.size
            high_conf_pixels = high_conf_mask.sum()
            high_conf_ratio = high_conf_pixels / total_pixels * 100
            mean_conf = conf_np.mean()
            median_conf = np.median(conf_np)
            
            stats_text = f'Statistics:\n'
            stats_text += f'Total pixels: {total_pixels}\n'
            stats_text += f'High conf pixels: {high_conf_pixels}\n'
            stats_text += f'High conf ratio: {high_conf_ratio:.2f}%\n'
            stats_text += f'Mean confidence: {mean_conf:.3f}\n'
            stats_text += f'Median confidence: {median_conf:.3f}\n'
            stats_text += f'Threshold: {confidence_threshold:.2f}'
            
            axes[1, 2].text(0.1, 0.5, stats_text, ha='left', va='center', 
                          transform=axes[1, 2].transAxes, fontsize=11,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(frame_dir, f"{face_key}_confidence_visualization.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # 单独保存各个图像
            # 保存原始RGB
            rgb_uint8 = (rgb_np * 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(frame_dir, f"{face_key}_rgb.png"), rgb_bgr)
            
            # 保存深度图
            if depth_np is not None:
                depth_uint8 = (depth_vis * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(frame_dir, f"{face_key}_depth.png"), depth_uint8)
                
                # 保存过滤后的深度图
                filtered_depth_uint8 = (filtered_depth_vis * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(frame_dir, f"{face_key}_depth_filtered.png"), filtered_depth_uint8)
            
            # 保存置信度图
            conf_uint8 = (conf_np * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(frame_dir, f"{face_key}_confidence.png"), conf_uint8)
            
            # 保存RGB+Mask
            rgb_with_mask_uint8 = (rgb_with_mask * 255).astype(np.uint8)
            rgb_with_mask_bgr = cv2.cvtColor(rgb_with_mask_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(frame_dir, f"{face_key}_rgb_with_mask.png"), rgb_with_mask_bgr)
        
        print(f"Saved initialization confidence visualization for frame {cur_frame_idx} to {frame_dir}")
        
    except Exception as e:
        print(f"Warning: Failed to save initialization confidence visualization: {e}")

def save_depth_comparison(render_depth, estimated_depth, mask, cur_frame_idx, save_dir, scale_factor=1.0, rgb_image=None):
    """
    保存深度相关图像到一张图片中（front面，mask下）
    
    Args:
        render_depth: 渲染深度（tensor，形状为 (1, H, W) 或 (H, W)）
        estimated_depth: 预测深度（tensor，形状为 (1, H, W) 或 (H, W)）
        mask: 有效mask（tensor，形状为 (1, H, W) 或 (H, W)）
        cur_frame_idx: 当前帧索引
        save_dir: 保存目录
        scale_factor: 尺度对齐因子（保留参数以兼容，但不再使用）
        rgb_image: 原始RGB图像（tensor，形状为 (C, H, W) 或 (3, H, W)），可选
    """
    try:
        # 确保是numpy数组格式
        if isinstance(render_depth, torch.Tensor):
            render_depth_np = render_depth.detach().cpu().squeeze().numpy()
        else:
            render_depth_np = np.array(render_depth).squeeze()
        
        if isinstance(estimated_depth, torch.Tensor):
            estimated_depth_np = estimated_depth.detach().cpu().squeeze().numpy()
        else:
            estimated_depth_np = np.array(estimated_depth).squeeze()
        
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().squeeze().numpy().astype(bool)
        else:
            mask_np = np.array(mask).squeeze().astype(bool)
        
        # 应用mask
        render_depth_masked = np.where(mask_np, render_depth_np, np.nan)
        estimated_depth_masked = np.where(mask_np, estimated_depth_np, np.nan)
        
        # 创建保存目录
        comparison_dir = os.path.join(save_dir, "depth_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # 处理RGB图像
        rgb_np = None
        rgb_masked = None
        if rgb_image is not None:
            if isinstance(rgb_image, torch.Tensor):
                rgb_np = rgb_image.detach().cpu().permute(1, 2, 0).numpy()  # (H, W, C)
            else:
                rgb_np = np.array(rgb_image)
                if rgb_np.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                    rgb_np = rgb_np.transpose(1, 2, 0)
            rgb_np = np.clip(rgb_np, 0.0, 1.0)
            # RGB图像（masked）- 只显示mask区域内的RGB
            rgb_masked = rgb_np.copy()
            rgb_masked[~mask_np] = 0  # mask外区域设为黑色
        
        # 计算深度范围
        valid_render = render_depth_masked[~np.isnan(render_depth_masked)]
        valid_estimated = estimated_depth_masked[~np.isnan(estimated_depth_masked)]
        if len(valid_render) > 0:
            vmin_render = valid_render.min()
            vmax_render = valid_render.max()
        else:
            vmin_render, vmax_render = 0, 10
        if len(valid_estimated) > 0:
            vmin_estimated = valid_estimated.min()
            vmax_estimated = valid_estimated.max()
        else:
            vmin_estimated, vmax_estimated = 0, 10
        
        # 创建2x2或2x3的布局
        if rgb_image is not None:
            fig, axes = plt.subplots(2, 2, figsize=(14, 14))
            fig.suptitle(f'Depth Comparison (Frame {cur_frame_idx})', fontsize=14)
            
            # 1. 原始RGB图像
            axes[0, 0].imshow(rgb_np)
            axes[0, 0].set_title('Original RGB Image', fontsize=12)
            axes[0, 0].axis('off')
            
            # 2. RGB图像（masked）
            axes[0, 1].imshow(rgb_masked)
            axes[0, 1].set_title('RGB Image (Masked)', fontsize=12)
            axes[0, 1].axis('off')
            
            # 3. 渲染深度（masked）
            im1 = axes[1, 0].imshow(render_depth_masked, cmap='viridis', vmin=vmin_render, vmax=vmax_render)
            axes[1, 0].set_title('Render Depth (Masked)', fontsize=12)
            axes[1, 0].axis('off')
            plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04, label='Depth (m)')
            
            # 4. 预测深度（masked）
            im2 = axes[1, 1].imshow(estimated_depth_masked, cmap='viridis', vmin=vmin_estimated, vmax=vmax_estimated)
            axes[1, 1].set_title('Estimated Depth (Masked)', fontsize=12)
            axes[1, 1].axis('off')
            plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04, label='Depth (m)')
        else:
            # 如果没有RGB图像，只显示深度图
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            fig.suptitle(f'Depth Comparison (Frame {cur_frame_idx})', fontsize=14)
            
            # 1. 渲染深度（masked）
            im1 = axes[0].imshow(render_depth_masked, cmap='viridis', vmin=vmin_render, vmax=vmax_render)
            axes[0].set_title('Render Depth (Masked)', fontsize=12)
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Depth (m)')
            
            # 2. 预测深度（masked）
            im2 = axes[1].imshow(estimated_depth_masked, cmap='viridis', vmin=vmin_estimated, vmax=vmax_estimated)
            axes[1].set_title('Estimated Depth (Masked)', fontsize=12)
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Depth (m)')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(comparison_dir, f"depth_comparison_frame_{cur_frame_idx:05d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        Log(f"Warning: Failed to save depth comparison: {e}")


def save_tracking_comparison(render_pkgs, viewpoint, cur_frame_idx, tracking_itr, save_dir):
    """
    保存跟踪过程中的渲染图像和真实图像对比
    上方显示所有渲染面，下方显示所有真实面

    Args:
        render_pkgs: 渲染结果字典，包含各个面的渲染图像
        viewpoint: 当前视角
        cur_frame_idx: 当前帧索引
        tracking_itr: 跟踪迭代次数
        save_dir: 保存目录
    """
    try:
        # 创建保存目录
        comparison_dir = os.path.join(save_dir, "tracking_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # 为每次保存创建独立的子文件夹
        frame_dir = os.path.join(comparison_dir, f"frame_{cur_frame_idx:06d}_itr_{tracking_itr:03d}")
        os.makedirs(frame_dir, exist_ok=True)
        
        # 存储每个面的PSNR值
        psnr_dict = {}

        # 定义面的顺序（立方体贴图的4个面）
        face_order = viewpoint.keep_faces
        num_faces = len(face_order)

        # 根据面的数量自适应子图布局
        # 每个面需要显示两张图（渲染图和真实图），所以是2行
        # 列数等于面的数量
        nrows = 2
        ncols = num_faces
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 8))
        
        # 确保axes是二维数组（当ncols=1时，plt.subplots返回一维数组，需要reshape）
        if axes.ndim == 1:
            axes = axes.reshape(nrows, ncols)

        # 处理每个面
        for i, face_key in enumerate(face_order):
            if face_key in render_pkgs:
                render_pkg = render_pkgs[face_key]
                tensor_render = render_pkg["render"]  # 渲染图像
                tensor_gt = viewpoint.Cubemap_image[face_key]  # 真实图像

                # 处理渲染图像（上方）
                if isinstance(tensor_render, torch.Tensor):
                    if len(tensor_render.shape) == 4:  # [B, C, H, W]
                        tensor_render = tensor_render.squeeze(0)
                    if tensor_render.shape[0] == 3:  # [C, H, W]
                        tensor_render = tensor_render.permute(1, 2, 0)
                    render_np = tensor_render.cpu().detach().numpy()
                else:
                    render_np = tensor_render

                # 处理真实图像（下方）
                if isinstance(tensor_gt, torch.Tensor):
                    if len(tensor_gt.shape) == 4:  # [B, C, H, W]
                        tensor_gt = tensor_gt.squeeze(0)
                    if tensor_gt.shape[0] == 3:  # [C, H, W]
                        tensor_gt = tensor_gt.permute(1, 2, 0)
                    gt_np = tensor_gt.cpu().detach().numpy()
                else:
                    gt_np = tensor_gt

                # 确保像素值在合理范围内 [0, 1]
                def normalize_image(img):
                    """将图像像素值归一化到[0,1]范围"""
                    if img is None:
                        return None
                    
                    try:
                        # 检查数据类型和范围
                        if img.dtype == np.uint8:
                            # 如果是uint8，转换为float并归一化
                            img_normalized = img.astype(np.float32) / 255.0
                        elif img.dtype == np.uint16:
                            # 如果是uint16，转换为float并归一化
                            img_normalized = img.astype(np.float32) / 65535.0
                        elif img.dtype in [np.float32, np.float64]:
                            # 如果是float类型，直接复制
                            img_normalized = img.copy().astype(np.float32)
                        else:
                            # 其他类型，尝试转换为float32
                            img_normalized = img.astype(np.float32)
                        
                        # 检查是否有超出[0,1]范围的值
                        min_val = img_normalized.min()
                        max_val = img_normalized.max()
                        
                        # 处理特殊情况
                        if np.isnan(min_val) or np.isnan(max_val):
                            print(f"警告: 图像包含NaN值，使用零填充")
                            img_normalized = np.nan_to_num(img_normalized, nan=0.0)
                            min_val, max_val = 0.0, 1.0
                        
                        if np.isinf(min_val) or np.isinf(max_val):
                            print(f"警告: 图像包含无穷值，进行裁剪")
                            img_normalized = np.clip(img_normalized, -1e6, 1e6)
                            min_val, max_val = img_normalized.min(), img_normalized.max()
                        
                        # 如果值范围异常，进行归一化
                        if min_val < 0.0 or max_val > 1.0:
                            if max_val > min_val:  # 避免除零
                                # 线性映射到[0,1]
                                img_normalized = (img_normalized - min_val) / (max_val - min_val)
                            else:
                                # 如果所有值相同，设为0.5
                                img_normalized = np.full_like(img_normalized, 0.5)
                        
                        # 最终裁剪确保在[0,1]范围内
                        img_normalized = np.clip(img_normalized, 0.0, 1.0)
                        
                        return img_normalized
                        
                    except Exception as e:
                        print(f"错误: 图像归一化失败: {e}")
                        # 返回一个默认的黑色图像
                        if len(img.shape) == 3:
                            return np.zeros_like(img, dtype=np.float32)
                        else:
                            return np.zeros((100, 100, 3), dtype=np.float32)

                # 归一化图像
                render_np = normalize_image(render_np)
                gt_np = normalize_image(gt_np)

                # 验证归一化后的图像
                if render_np is not None:
                    render_min, render_max = render_np.min(), render_np.max()
                    if render_min < 0.0 or render_max > 1.0:
                        print(f"警告: 渲染图像 {face_key} 归一化后仍有超出[0,1]范围的值: [{render_min:.3f}, {render_max:.3f}]")
                
                if gt_np is not None:
                    gt_min, gt_max = gt_np.min(), gt_np.max()
                    if gt_min < 0.0 or gt_max > 1.0:
                        print(f"警告: 真实图像 {face_key} 归一化后仍有超出[0,1]范围的值: [{gt_min:.3f}, {gt_max:.3f}]")

                # 保存每个面的独立图像
                if render_np is not None and gt_np is not None:
                    # 将归一化后的图像转换为uint8格式保存
                    render_uint8 = (render_np * 255).astype(np.uint8)
                    gt_uint8 = (gt_np * 255).astype(np.uint8)
                    
                    # 转换BGR格式（cv2使用BGR）
                    render_bgr = cv2.cvtColor(render_uint8, cv2.COLOR_RGB2BGR)
                    gt_bgr = cv2.cvtColor(gt_uint8, cv2.COLOR_RGB2BGR)
                    
                    # 保存渲染图像
                    render_path = os.path.join(frame_dir, f"{face_key}_render.png")
                    cv2.imwrite(render_path, render_bgr)
                    
                    # 保存真实图像
                    gt_path = os.path.join(frame_dir, f"{face_key}_gt.png")
                    cv2.imwrite(gt_path, gt_bgr)
                    
                    # 计算PSNR
                    # 需要将numpy数组转换回torch tensor，并确保格式为[C, H, W]
                    # 使用contiguous()确保tensor在内存中是连续的，避免view()错误
                    render_tensor = torch.from_numpy(render_np).permute(2, 0, 1).unsqueeze(0).contiguous()  # [1, C, H, W]
                    gt_tensor = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).contiguous()  # [1, C, H, W]
                    
                    # 计算PSNR（PSNR函数期望输入在[0,1]范围内）
                    try:
                        psnr_score = psnr(render_tensor, gt_tensor)
                        psnr_value = psnr_score.item()
                        psnr_dict[face_key] = psnr_value
                    except Exception as e:
                        print(f"警告: 计算 {face_key} 的PSNR时出错: {e}")
                        psnr_dict[face_key] = None

                # 绘制渲染图像（第一行）
                if render_np is not None:
                    axes[0, i].imshow(render_np, vmin=0.0, vmax=1.0)
                    axes[0, i].set_title(f'Rendered {face_key}', fontsize=10)
                else:
                    axes[0, i].text(0.5, 0.5, f'No render {face_key}', ha='center', va='center', transform=axes[0, i].transAxes)
                axes[0, i].axis('off')

                # 绘制真实图像（第二行）
                if gt_np is not None:
                    axes[1, i].imshow(gt_np, vmin=0.0, vmax=1.0)
                    axes[1, i].set_title(f'GT {face_key}', fontsize=10)
                else:
                    axes[1, i].text(0.5, 0.5, f'No GT {face_key}', ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].axis('off')
            else:
                # 如果某个面不存在，显示空白
                axes[0, i].text(0.5, 0.5, f'No {face_key}', ha='center', va='center', transform=axes[0, i].transAxes)
                axes[0, i].axis('off')
                axes[1, i].text(0.5, 0.5, f'No {face_key}', ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].axis('off')

        # 调整布局
        plt.tight_layout()

        # 保存图像
        filename = f"frame_{cur_frame_idx:06d}_itr_{tracking_itr:03d}_all_faces.png"
        filepath = os.path.join(comparison_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)  # 关闭图形以释放内存

        # 保存PSNR到txt文件
        psnr_filepath = os.path.join(frame_dir, "psnr.txt")
        with open(psnr_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Frame: {cur_frame_idx}, Iteration: {tracking_itr}\n")
            f.write("=" * 50 + "\n")
            f.write(f"{'Face':<15} {'PSNR (dB)':<15}\n")
            f.write("-" * 50 + "\n")
            
            total_psnr = 0
            valid_count = 0
            for face_key in face_order:
                if face_key in psnr_dict:
                    psnr_value = psnr_dict[face_key]
                    if psnr_value is not None:
                        f.write(f"{face_key:<15} {psnr_value:.4f}\n")
                        total_psnr += psnr_value
                        valid_count += 1
                    else:
                        f.write(f"{face_key:<15} {'N/A':<15}\n")
                else:
                    f.write(f"{face_key:<15} {'N/A':<15}\n")
            
            f.write("-" * 50 + "\n")
            if valid_count > 0:
                mean_psnr = total_psnr / valid_count
                f.write(f"{'Mean PSNR':<15} {mean_psnr:.4f}\n")
            else:
                f.write(f"{'Mean PSNR':<15} {'N/A':<15}\n")

        print(f"Saved tracking comparison image for frame {cur_frame_idx}, iteration {tracking_itr}")
        print(f"Saved individual face images and PSNR to {frame_dir}")

    except Exception as e:
        print(f"Error saving tracking comparison: {e}")
        import traceback
        traceback.print_exc()
class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False # 初始化状态，`False` 表示前端 SLAM 还未初始化
        self.kf_indices = [] # 存储关键帧索引
        self.monocular = config["Training"]["monocular"] # 是否为单目
        self.iteration_count = 0
        self.occ_aware_visibility = {} # 遮挡信息
        self.current_window = [] # 关键帧窗口

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1
        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False
        self.submap_manager = None  # 子图管理器（将从主进程传递）

        self.depth_replace_cfg = self.config.get("DepthReplacement", {})
        self.depth_replacement_enabled = self.depth_replace_cfg.get("enabled", False)
        self.face_depth_history = {}
        self.face_scale_history = {}
        
        # ========== 可选功能：计算tracking过程中的front面PSNR ==========
        # 如果需要计算tracking过程中的PSNR，取消下面的注释
        # 这将计算每一帧的front面PSNR，并在SLAM结束时计算平均值保存到results.txt
        self.tracking_psnr_list = []  # 存储每帧的front面PSNR
        
        # ========== 帧范围控制 ==========
        # 从配置中读取起始帧和结束帧（可选）
        dataset_cfg = self.config.get("Dataset", {})
        self.start_frame = dataset_cfg.get("start_frame", None)  # 起始帧索引，None表示从0开始
        self.end_frame = dataset_cfg.get("end_frame", None)  # 结束帧索引，None表示处理到数据集末尾
        # ================================================================

    # 该方法用于设置 SLAM 前端（FrontEnd）的超参数
    def set_hyperparams(self):
        # 结果保存相关参数
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]
        self.make_video = self.config["Results"].get("make_video", False)

        # 训练超参数
        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    # 添加新的关键帧
    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"] # RGB边界阈值
        self.kf_indices.append(cur_frame_idx) # 将当前帧索引添加到关键帧索引
        # 获取相机：如果cameras中没有，说明是在初始化时被清空了，需要从传入的viewpoint获取
        if cur_frame_idx in self.cameras:
            viewpoint = self.cameras[cur_frame_idx]
        else:
            # 如果cameras中没有，说明是在初始化时，viewpoint应该已经传入
            # 这种情况下，我们需要从initialize方法传入的viewpoint获取
            # 但这里没有viewpoint参数，所以需要确保cameras中有
            # 实际上，在initialize中，viewpoint已经在之前被添加到cameras中了
            # 如果这里没有，说明有问题
            raise KeyError(f"Camera {cur_frame_idx} not found in cameras dict. This should not happen during initialization.")
        gt_img_dict = {key: img.cuda() for key, img in viewpoint.Cubemap_image.items()}  # 获取多面图像
        initial_depth_dict = {}  # 保存每个方向的初始深度
        # 是否使用深度预测（仅对ERP数据集生效），默认开启
        use_depth_prediction = self.config.get("Dataset", {}).get("use_depth_prediction", True)

        # 如果是ERP数据集且配置启用深度预测，进行深度估计
        if viewpoint.dataset_type == "ERP" and use_depth_prediction:
            # 估计深度（ERP格式）
            depth_erp = estimate_depth(viewpoint.path)
            depth_tensor = torch.from_numpy(depth_erp).float().unsqueeze(0).to(self.device)
            estimated_depth_dict = self.Cube.convert(depth_tensor)  # 得到各面的估计深度字典
            
            # 保存原始渲染深度（如果存在）
            render_depth_dict = depth.copy() if depth is not None and len(depth) > 0 else None
            
            # 初始化depth_mask字典，用于保存每个面的有效深度mask
            if not hasattr(viewpoint, 'depth_mask') or viewpoint.depth_mask is None:
                viewpoint.depth_mask = {}
            
            # 不再进行深度对齐，直接使用预测深度（不管是否初始化）
            depth = {}
            for key, estimated_depth_face in estimated_depth_dict.items():
                if not isinstance(estimated_depth_face, torch.Tensor):
                    estimated_depth_face = torch.from_numpy(estimated_depth_face).float().to(self.device)
                if estimated_depth_face.dim() == 2:
                    estimated_depth_face = estimated_depth_face.unsqueeze(0)
                
                # 通过预测深度生成mask（不进行对齐）
                # 深度在 (0.01, 100.0) 范围内视为有效，>100 视为天空等远处区域并被mask掉
                valid_mask = (estimated_depth_face < 100.0) & (estimated_depth_face > 0.01)
                
                # 保存mask到viewpoint，用于loss计算
                viewpoint.depth_mask[key] = valid_mask.detach().clone()
                
                # mask内使用预测深度，mask外设置为0（无效深度，不生成高斯点）
                # 不管是否初始化，都使用预测深度
                final_depth = torch.where(valid_mask, estimated_depth_face, torch.zeros_like(estimated_depth_face))
                depth[key] = final_depth
                
                # 如果是front面且需要保存结果，生成深度对比图
                if key == "front" and self.save_results and hasattr(self, 'save_dir'):
                    # 获取front面的RGB图像
                    rgb_front = viewpoint.Cubemap_image.get("front", None)
                    # 如果有渲染深度，用于可视化对比
                    render_depth_face = None
                    if render_depth_dict is not None and len(render_depth_dict) > 0 and key in render_depth_dict:
                        render_depth_face = render_depth_dict[key]
                        if not isinstance(render_depth_face, torch.Tensor):
                            render_depth_face = torch.from_numpy(render_depth_face).float().to(self.device)
                        if render_depth_face.dim() == 2:
                            render_depth_face = render_depth_face.unsqueeze(0)
                    # save_depth_comparison(
                    #     render_depth_face,
                    #     estimated_depth_face,
                    #     valid_mask,
                    #     cur_frame_idx,
                    #     self.save_dir,
                    #     scale_factor=1.0,
                    #     rgb_image=rgb_front
                    # )
        
        for key, gt_img in gt_img_dict.items():
            # 每个面的RGB有效像素掩码
            valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]

            if self.monocular:
                if depth is not None and key in depth:
                    depth_img = depth[key].detach().clone()
                    opacity_img = opacity[key].detach() if opacity is not None and key in opacity else None
                    
                    # 记录原始无效深度区域（深度为0的区域，这些区域不应该生成高斯点）
                    invalid_depth_region = (depth_img <= 0)
                    
                    # 如果有opacity信息，进行深度修正
                    if opacity_img is not None:
                        median_depth, std, valid_mask = get_median_depth(
                            depth_img, opacity_img, mask=valid_rgb, return_std=True
                        )
                        invalid_depth_mask = torch.logical_or(
                            depth_img > median_depth + std, depth_img < median_depth - std
                        )
                        invalid_depth_mask = torch.logical_or(invalid_depth_mask, ~valid_mask)
                        # 只修正有效深度区域内的异常值，保持无效区域为0
                        depth_img[invalid_depth_mask & ~invalid_depth_region] = median_depth
                    
                    initial_depth = depth_img
                    # 将无效RGB区域设置为0（无效深度，不生成高斯点）
                    initial_depth[~valid_rgb] = 0
                    # 确保原始无效深度区域保持为0（不生成高斯点）
                    initial_depth[invalid_depth_region] = 0
                    # 确保深度为0或负数的区域都被标记为无效（不生成高斯点）
                    initial_depth[initial_depth <= 0] = 0
                    depth_np = initial_depth.squeeze(0).detach().cpu().numpy()
                    initial_depth_dict[key] = depth_np
                else:
                    # 没有深度信息，生成随机深度
                    initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2], device=gt_img.device)
                    initial_depth += torch.randn_like(initial_depth) * 0.3
                    depth_np = initial_depth.squeeze(0).detach().cpu().numpy()
                    initial_depth_dict[key] = depth_np
            else:
                # Stereo模式下直接退出，不处理深度
                return initial_depth_dict
        
        return initial_depth_dict

    # SLAM初始化
    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = [] # 关键帧索引
        self.iteration_count = 0 # 重置SLAM运行的迭代次数
        self.occ_aware_visibility = {} # 清空遮挡信息
        self.current_window = [] # 清空BA滑动窗口
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map_dict = self.add_new_keyframe(cur_frame_idx, init=True) # 创建新的关键帧
        self.request_init(cur_frame_idx, viewpoint, depth_map_dict) # 请求SLAM初始化
        self.reset = False # 标记初始化完成


    def tracking(self, cur_frame_idx, viewpoint):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        viewpoint.update_RT(prev.R, prev.T)

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)

        start_time = time.time()
        for tracking_itr in range(self.tracking_itr_num):
            total_loss = 0
            render_pkgs = {}

            for face_key in viewpoint.Cubemap_image:
                render_pkg = render( viewpoint, self.gaussians, self.pipeline_params, self.background, face_key=face_key, use_quantized=True )

                image = render_pkg["render"]
                depth = render_pkg["depth"]
                opacity = render_pkg["opacity"]

                loss_tracking = get_loss_tracking( self.config, image, depth, opacity, viewpoint, face_key=face_key )

                total_loss += loss_tracking
                render_pkgs[face_key] = render_pkg

            pose_optimizer.zero_grad()
            total_loss.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.Cubemap_image['front'],
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break

        # 计算中值深度（在循环结束后）
        depths = []
        for face_key in viewpoint.Cubemap_image:
            depth = render_pkgs[face_key]['depth']
            opacity = render_pkgs[face_key]['opacity']
            median = get_median_depth(depth, opacity)
            depths.append(median)
        self.median_depth = sum(depths) / len(depths)
        loop_time = time.time() - start_time
        # 计算实际迭代次数（tracking_itr从0开始，所以实际次数是tracking_itr+1）
        actual_iterations = tracking_itr + 1
        # 避免除以零的情况
        if actual_iterations > 0:
            average_time = (loop_time / actual_iterations) * 1000
        else:
            average_time = 0.0
        print(
            f'Frame:{cur_frame_idx} |'
            f'tracking_itr:{actual_iterations} |'
            f'Time:{loop_time:.2f}s |'
            f'average_time:{average_time:.1f}ms |'
            f'Loss:{total_loss.item():.4f} |'
        #     # f'converged:{converged} |'
        #     #f'path:{viewpoint.path}'
        )
        # 实时录制视频：
        # 1) 使用当前高斯点云 + 固定位姿渲染一个固定视角（保存在 video/fixed/）
        # 2) 同时保存当前tracking下的 front/left/right/back 渲染结果（保存在 video/<face>/）
        if getattr(self, "make_video", False) and hasattr(self, "save_dir") and self.save_dir is not None:
            save_video_frame_realtime(
                viewpoint,
                self.gaussians,
                self.pipeline_params,
                self.background,
                self.save_dir,
                cur_frame_idx,
            )
            # 保存实时的各面渲染结果
            for face_key in ["front", "left", "right", "back"]:
                if face_key in render_pkgs:
                    save_face_render_realtime(
                        render_pkgs[face_key]["render"],
                        self.save_dir,
                        cur_frame_idx,
                        face_key,
                    )
        
        # ========== 可选功能：计算front面的PSNR ==========
        # 如果需要计算tracking过程中的PSNR，取消下面的注释
        if hasattr(self, 'tracking_psnr_list') and 'front' in render_pkgs and 'front' in viewpoint.Cubemap_image:
            try:
                render_pkg_front = render_pkgs['front']
                tensor_render = render_pkg_front["render"]  # 渲染图像
                tensor_gt = viewpoint.Cubemap_image['front']  # 真实图像
                
                # 处理渲染图像
                if isinstance(tensor_render, torch.Tensor):
                    if len(tensor_render.shape) == 4:  # [B, C, H, W]
                        tensor_render = tensor_render.squeeze(0)
                    if tensor_render.shape[0] == 3:  # [C, H, W]
                        tensor_render = tensor_render.permute(1, 2, 0)
                    render_np = tensor_render.cpu().detach().numpy()
                else:
                    render_np = tensor_render
                
                # 处理真实图像
                if isinstance(tensor_gt, torch.Tensor):
                    if len(tensor_gt.shape) == 4:  # [B, C, H, W]
                        tensor_gt = tensor_gt.squeeze(0)
                    if tensor_gt.shape[0] == 3:  # [C, H, W]
                        tensor_gt = tensor_gt.permute(1, 2, 0)
                    gt_np = tensor_gt.cpu().detach().numpy()
                else:
                    gt_np = tensor_gt
                
                # 归一化图像到[0,1]范围
                def normalize_for_psnr(img):
                    if img is None:
                        return None
                    if img.dtype == np.uint8:
                        img = img.astype(np.float32) / 255.0
                    elif img.dtype == np.uint16:
                        img = img.astype(np.float32) / 65535.0
                    elif img.dtype in [np.float32, np.float64]:
                        img = img.astype(np.float32)
                    else:
                        img = img.astype(np.float32)
                    
                    # 确保在[0,1]范围内
                    min_val = img.min()
                    max_val = img.max()
                    if min_val < 0.0 or max_val > 1.0:
                        if max_val > min_val:
                            img = (img - min_val) / (max_val - min_val)
                    img = np.clip(img, 0.0, 1.0)
                    return img
                
                render_np = normalize_for_psnr(render_np)
                gt_np = normalize_for_psnr(gt_np)
                
                if render_np is not None and gt_np is not None:
                    # 转换为torch tensor格式 [1, C, H, W]
                    render_tensor = torch.from_numpy(render_np).permute(2, 0, 1).unsqueeze(0).contiguous()
                    gt_tensor = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).contiguous()
                    
                    # 计算PSNR
                    psnr_score = psnr(render_tensor, gt_tensor)
                    psnr_value = psnr_score.item()
                    self.tracking_psnr_list.append(psnr_value)
            except Exception as e:
                # 如果计算失败，跳过这一帧
                pass
        # ================================================================

        return render_pkgs

    # 该方法用于 判断当前帧是否应作为关键帧（Keyframe），通过位姿变化（平移）+ 视图重叠度 进行决策
    def is_keyframe(
        self,
        cur_frame_idx, # 帧ID
        last_keyframe_idx, # 上一帧 ID
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        # kf_translation 关键帧最大位移阈值（相机平移量超过 kf_translation * median_depth，则插入关键帧
        kf_translation = self.config["Training"]["kf_translation"]
        # 关键帧最小位移阈值（相机平移量超过 kf_min_translation * median_depth，且视图重叠度小于 kf_overlap，则插入关键帧）
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        # 关键帧的最小重叠比（小于该值则插入关键帧）
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx] # 当前帧
        last_kf = self.cameras[last_keyframe_idx] # 上一帧
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T) # 当前帧 世界坐标系 → 相机坐标系 变换矩阵
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T) # 上一关键帧 世界坐标系 → 相机坐标系 变换矩阵
        last_kf_WC = torch.linalg.inv(last_kf_CW) # 上一关键帧 相机坐标系 → 世界坐标系 变换矩阵（求逆）
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3]) # 计算 当前帧与上一关键帧之间的平移距离，欧几里的距离
        dist_check = dist > kf_translation * self.median_depth # 相机移动较大，则插入关键帧
        dist_check2 = dist > kf_min_translation * self.median_depth # 重叠率较低，则插入关键帧

        # 计算视图重叠度
        union = torch.logical_or( # 计算当前帧和上一关键帧可见点的并集数量
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and( # 计算当前帧和上一关键帧可见点的交集数量
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union # 交并比 = intersection / union，表示 两帧的可见点重叠程度
        # 插入条件：1.视图变化较大（平移距离大或者重叠小）2.平移变化大（相机移动大）
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)): # 开始两个关键帧不会被检查
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and( # 当前帧和kf_idx帧的交集
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min( # 取可见数量较小者
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom # 值越大表示重叠越高


            cut_off = ( # 设定一个重叠阈值，若小于则添加到 to_remove 中
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove: # 最后一个帧被剔除
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]

        # 当前帧在世界坐标的位姿
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        # 窗口大小超过窗口上限时
        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame) # 移出与其他关键帧距离最远的关键帧

        return window, removed_frame # 返回更新的关键帧窗口和移除的关键帧索引

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map_dict):
        msg = ["init", cur_frame_idx, viewpoint, depth_map_dict]
        self.backend_queue.put(msg)
        self.requested_init = True

    # 传递相关数据到前端
    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            # 检查关键帧是否存在于cameras中（可能在清空GPU时被清除了）
            if kf_id in self.cameras:
                self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())
            else:
                # 如果关键帧不存在，记录警告但继续执行（可能在子图切换时被清除了）
                Log(f"Warning: Keyframe {kf_id} not found in cameras, skipping pose update", tag="Submap")

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()


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
        
        # 在子进程中重新初始化子图管理器（如果配置了且未传递）
        # 因为多进程环境下对象无法直接传递，需要在子进程中重新创建
        # 注意：需要先调用set_hyperparams()来设置save_dir
        self.set_hyperparams()
        if self.submap_manager is None:
            submap_config = self.config.get("Submap", {})
            if submap_config.get("enabled", True):
                from utils.submap_manager import SubmapManager
                # 使用前端的save_dir（在set_hyperparams中设置）
                save_dir = getattr(self, 'save_dir', None)
                self.submap_manager = SubmapManager(self.config, save_dir=save_dir)
                # 关联frontend引用
                self.submap_manager.frontend = self
                Log(f"SubmapManager re-initialized in frontend process (save_dir={save_dir})", tag="Submap")
        
        # 设置起始帧
        cur_frame_idx = self.start_frame if self.start_frame is not None else 0
        # 确保起始帧在有效范围内
        if cur_frame_idx < 0:
            cur_frame_idx = 0
        if cur_frame_idx >= len(self.dataset):
            Log(f"Warning: start_frame ({cur_frame_idx}) >= dataset length ({len(self.dataset)}), using 0 instead")
            cur_frame_idx = 0
        
        # 设置结束帧
        end_frame_idx = self.end_frame if self.end_frame is not None else len(self.dataset)
        # 确保结束帧在有效范围内
        if end_frame_idx > len(self.dataset):
            end_frame_idx = len(self.dataset)
        if end_frame_idx <= cur_frame_idx:
            Log(f"Warning: end_frame ({end_frame_idx}) <= start_frame ({cur_frame_idx}), using dataset length ({len(self.dataset)}) instead")
            end_frame_idx = len(self.dataset)
        
        Log(f"Processing frames from {cur_frame_idx} to {end_frame_idx} (total: {len(self.dataset)})")
        # 计算投影矩阵
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx_face,
            cy=self.dataset.cy_face,
            W=self.dataset.wCube,
            H=self.dataset.hCube,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            # 如果 frontend_queue 为空，说明没有后端数据需要处理
            if self.frontend_queue.empty():
                tic.record()
                # 终止条件：达到结束帧或数据集帧用完，结束SLAM
                if cur_frame_idx >= end_frame_idx or cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                # 单线程下，等待关键帧完成后再继续
                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                # 创建相机 viewpoint，从中读取 cur_frame_idx 对应的帧
                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix, self.Cube
                )
                # 计算梯度遮罩（compute_grad_mask），用于优化跟踪效果，更新了 grad_mask
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint
                
                # 子图管理：判断是否需要创建新子图
                if hasattr(self, 'submap_manager') and self.submap_manager is not None:
                    self.submap_manager.increment_frame_count()
                    if self.submap_manager.should_start_new_submap(cur_frame_idx):
                        # 在创建新子图前，如果有旧子图需要保存，直接在前端保存
                        old_submap_id = self.submap_manager.current_submap_id
                        if old_submap_id is not None:
                            old_submap = self.submap_manager.get_submap(old_submap_id)
                            if old_submap is not None and not old_submap.is_saved and old_submap.keyframes:
                                # 先finalize子图
                                self.submap_manager._finalize_current_submap()
                                
                                # 尝试从backend同步最新的gaussians（非阻塞）
                                # 检查frontend_queue中是否有最新的gaussians数据
                                import queue
                                try:
                                    # 非阻塞获取最新的gaussians数据
                                    while True:
                                        try:
                                            data = self.frontend_queue.get_nowait()
                                            if data[0] == "sync_backend" or data[0] == "keyframe":
                                                self.sync_backend(data)
                                                Log(f"Synced latest gaussians from backend before saving submap {old_submap_id}", tag="Submap")
                                        except queue.Empty:
                                            break
                                except:
                                    pass
                                
                                # 请求backend推送最新的gaussians
                                if self.backend_queue is not None:
                                    self.backend_queue.put(["push_gaussians"])
                                    # 等待backend响应（最多等待200ms）
                                    max_wait = 0.2
                                    wait_interval = 0.01
                                    waited = 0.0
                                    synced = False
                                    while waited < max_wait:
                                        try:
                                            data = self.frontend_queue.get_nowait()
                                            if data[0] == "sync_backend" or data[0] == "keyframe":
                                                self.sync_backend(data)
                                                Log(f"Synced gaussians after requesting from backend", tag="Submap")
                                                synced = True
                                                break
                                        except queue.Empty:
                                            time.sleep(wait_interval)
                                            waited += wait_interval
                                    if not synced:
                                        Log(f"Warning: Did not receive gaussians from backend after request", tag="Submap")
                                
                                if self.gaussians is not None:
                                    Log(f"Saving submap {old_submap_id} in frontend before reset (keyframes: {len(old_submap.keyframes)})", tag="Submap")
                                    ckpt_path = self.submap_manager.save_dict_to_ckpt(
                                        old_submap_id,
                                        self.gaussians,
                                        self.cameras
                                    )
                                    if ckpt_path:
                                        Log(f"Successfully saved submap {old_submap_id} to {ckpt_path}", tag="Submap")
                                    else:
                                        Log(f"Failed to save submap {old_submap_id} - no gaussians found for keyframes {old_submap.keyframes[:5]}... (may have been pruned/reset)", tag="Submap")
                                else:
                                    Log(f"Warning: gaussians is None, cannot save submap {old_submap_id}", tag="Submap")
                        
                        new_submap_id = self.submap_manager.create_new_submap(cur_frame_idx)
                        # 子图完成后，需要重新初始化系统
                        self.reset = True
                        if old_submap_id is not None:
                            Log(f"Submap {old_submap_id} completed, resetting system for new submap {new_submap_id} at frame {cur_frame_idx}", tag="Submap")
                        else:
                            Log(f"Creating first submap {new_submap_id}, resetting system at frame {cur_frame_idx}", tag="Submap")

                if self.reset: # id = 0
                    self.initialize(cur_frame_idx, viewpoint) # 初始化，之后 self.reset = False
                    self.current_window.append(cur_frame_idx)
                    print('Length of dataset:', len(self.dataset))
                    cur_frame_idx += 1 # 处理下一帧
                    continue


                # 判断是否完成初始化，需要windows_size数量的关键帧
                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                # 视觉跟踪，返回 render_pkg，其中包含 depth、opacity、n_touched（可见点数）
                render_pkgs = self.tracking(cur_frame_idx, viewpoint)

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                # 发送数据（高斯点云数据和关键帧）给 GUI
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                # 如果请求了关键帧，则进行清理，并进入下一帧
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                # 检查当前帧是否应该作为关键帧：1.时间间隔是否超过kf_interval，2.当前帧是否有足够多的可见点curr_visibility，3.关键帧判断逻辑self.is_keyframe
                # 统一的关键帧判断逻辑（不再检查帧ID是否为5的倍数）
                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval

                total_n_touched = None
                for face_key in viewpoint.Cubemap_image:
                    n_touched = render_pkgs[face_key]["n_touched"]
                    if total_n_touched is None:
                        total_n_touched = n_touched.clone()
                    else:
                        total_n_touched += n_touched  # 逐元素累加被观测到的次数
                curr_visibility = total_n_touched.long()


                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )

                if len(self.current_window) < self.window_size: # 8
                    union = torch.logical_or( # 并集，当前帧和上一关键帧的可见点联合区域
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and( # 交集，当前帧和上一关键帧的可见点共同区域
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union # 交并比
                    create_kf = ( # 判断是否是关键帧
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                    if self.single_thread:
                        create_kf = check_time and create_kf
                
                # 添加关键帧
                if create_kf:
                    Log("create_kf", 'id', cur_frame_idx)
                    # 将当前帧加入关键帧窗口
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    # 计算深度图，并存储关键帧信息
                    depth_dict = {}
                    opacity_dict = {}
                    for face_key, render_pkg in render_pkgs.items():
                        depth_dict[face_key] = render_pkg['depth']
                        opacity_dict[face_key] = render_pkg['opacity']

                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=depth_dict,
                        opacity=opacity_dict,
                        init=False,
                    )
                    self.request_keyframe( # 向backend发送keyframe请求
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                    
                    # 子图管理：添加关键帧到当前子图
                    if hasattr(self, 'submap_manager') and self.submap_manager is not None:
                        self.submap_manager.add_keyframe_to_current_submap(cur_frame_idx)
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0 # 10的整数倍
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))


            # 接收后端的信息
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
