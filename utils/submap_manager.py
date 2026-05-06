"""
子图管理器模块
负责子图的创建、保存和管理
"""

import os
import torch
from typing import Optional, Dict, List
from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.logging_utils import Log


class Submap:
    """子图数据结构"""
    def __init__(self, submap_id: int):
        self.submap_id = submap_id
        self.start_frame_idx = None  # 子图起始帧索引
        self.end_frame_idx = None    # 子图结束帧索引
        self.keyframes = []           # 该子图包含的关键帧ID列表
        self.ckpt_path = None         # 保存路径
        self.is_active = True         # 是否为当前活跃子图
        self.is_saved = False         # 是否已保存到磁盘


class SubmapManager:
    """子图管理器"""
    
    def __init__(self, config, save_dir=None):
        """
        初始化子图管理器
        
        Args:
            config: 配置字典
            save_dir: 结果保存目录
        """
        self.config = config
        self.save_dir = save_dir
        
        # 子图管理
        self.submaps: Dict[int, Submap] = {}  # 子图字典
        self.current_submap_id: Optional[int] = None  # 当前活跃子图ID
        self.next_submap_id = 0  # 下一个子图ID
        
        # 帧计数（用于判断是否创建新子图）
        self.frame_count_since_last_submap = 0
        
        # 从配置读取参数
        submap_config = config.get("Submap", {})
        self.enabled = submap_config.get("enabled", True)
        self.frames_per_submap = submap_config.get("frames_per_submap", 200)
        self.submap_dir_name = submap_config.get("submap_dir", "submaps")
        self.save_keyframe_poses = submap_config.get("save_keyframe_poses", True)
        
        # 如果子图功能被禁用，直接返回
        if not self.enabled:
            Log("Submap feature is disabled", tag="Submap")
            return
        
        # 创建子图保存目录
        if save_dir:
            self.submap_dir = os.path.join(save_dir, self.submap_dir_name)
            os.makedirs(self.submap_dir, exist_ok=True)
            Log(f"Submap directory created: {self.submap_dir}", tag="Submap")
        else:
            self.submap_dir = None
        
        # 前后端引用（将在slam.py中设置）
        self.backend = None
        self.frontend = None
        
        Log(f"SubmapManager initialized: frames_per_submap={self.frames_per_submap}", tag="Submap")
    
    def should_start_new_submap(self, current_frame_idx: int) -> bool:
        """
        判断是否应该创建新子图
        
        Args:
            current_frame_idx: 当前帧索引
            
        Returns:
            bool: True表示需要创建新子图
        """
        if not self.enabled:
            return False
        
        # 情况1: 还没有创建过子图
        if self.current_submap_id is None:
            return True
        
        # 情况2: 基于帧数间隔判断
        if self.frame_count_since_last_submap >= self.frames_per_submap:
            return True
        
        # 情况3: 可以添加其他条件（后续扩展）
        # - 关键帧数量阈值
        # - 空间距离阈值
        # - 时间间隔阈值
        
        return False
    
    def create_new_submap(self, start_frame_idx: int) -> int:
        """
        创建新子图
        
        Args:
            start_frame_idx: 起始帧索引
            
        Returns:
            int: 新创建的子图ID
        """
        # 1. 如果存在当前子图，先完成它（标记为非活跃）
        old_submap_id = None
        if self.current_submap_id is not None:
            old_submap_id = self.current_submap_id
            self._finalize_current_submap()
        
        # 2. 分配新的子图ID
        submap_id = self.next_submap_id
        self.next_submap_id += 1
        
        # 3. 创建子图对象
        submap = Submap(submap_id)
        submap.start_frame_idx = start_frame_idx
        submap.end_frame_idx = start_frame_idx
        submap.is_active = True
        
        # 4. 添加到子图字典
        self.submaps[submap_id] = submap
        self.current_submap_id = submap_id
        
        # 5. 重置帧计数
        self.frame_count_since_last_submap = 0
        
        Log(f"Created new submap {submap_id} at frame {start_frame_idx}" + 
            (f" (previous submap {old_submap_id} finalized)" if old_submap_id is not None else ""), 
            tag="Submap")
        return submap_id
    
    def add_keyframe_to_current_submap(self, keyframe_idx: int):
        """
        添加关键帧到当前活跃子图
        
        Args:
            keyframe_idx: 关键帧索引
        """
        if not self.enabled:
            return
        
        if self.current_submap_id is None:
            # 如果还没有子图，先创建一个
            self.create_new_submap(keyframe_idx)
        
        submap = self.submaps[self.current_submap_id]
        if keyframe_idx not in submap.keyframes:
            submap.keyframes.append(keyframe_idx)
            submap.end_frame_idx = keyframe_idx
    
    def increment_frame_count(self):
        """增加帧计数"""
        if self.enabled:
            self.frame_count_since_last_submap += 1
    
    def _finalize_current_submap(self):
        """完成当前子图（标记为非活跃状态）"""
        if self.current_submap_id is not None:
            submap = self.submaps[self.current_submap_id]
            submap.is_active = False
            Log(f"Finalized submap {self.current_submap_id} (frames: {submap.start_frame_idx}-{submap.end_frame_idx}, keyframes: {len(submap.keyframes)})", tag="Submap")
    
    def _extract_submap_gaussians(self, gaussians: GaussianModel, 
                                  keyframes: List[int]) -> Dict:
        """
        从全局高斯模型中提取属于指定关键帧的高斯点数据
        
        Args:
            gaussians: 全局高斯模型
            keyframes: 关键帧ID列表
            
        Returns:
            dict: 提取的高斯点数据字典
        """
        # 创建掩码：高斯点的 unique_kfIDs 在 keyframes 中
        # 确保mask和设备与gaussians.unique_kfIDs一致
        device = gaussians.unique_kfIDs.device
        dtype = gaussians.unique_kfIDs.dtype
        mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device=device)
        for kf_idx in keyframes:
            # 将kf_idx转换为tensor并放到正确的设备上
            kf_idx_tensor = torch.tensor(kf_idx, dtype=dtype, device=device)
            mask |= (gaussians.unique_kfIDs == kf_idx_tensor)
        
        # 统计提取的高斯点数量
        num_points = mask.sum().item()
        if num_points == 0:
            Log(f"Warning: No gaussians found for keyframes {keyframes}", tag="Submap")
            return None
        
        # 提取高斯点数据（转换为CPU以节省GPU内存）
        extracted_data = {
            '_xyz': gaussians._xyz[mask].detach().cpu().clone(),
            '_features_dc': gaussians._features_dc[mask].detach().cpu().clone(),
            '_features_rest': gaussians._features_rest[mask].detach().cpu().clone(),
            '_opacity': gaussians._opacity[mask].detach().cpu().clone(),
            '_scaling': gaussians._scaling[mask].detach().cpu().clone(),
            '_rotation': gaussians._rotation[mask].detach().cpu().clone(),
            'unique_kfIDs': gaussians.unique_kfIDs[mask].detach().cpu().clone(),
            'n_obs': gaussians.n_obs[mask].detach().cpu().clone() if hasattr(gaussians, 'n_obs') else None,
            'max_radii2D': gaussians.max_radii2D[mask].detach().cpu().clone() if hasattr(gaussians, 'max_radii2D') else None,
        }
        
        Log(f"Extracted {num_points} gaussians for submap (keyframes: {keyframes})", tag="Submap")
        return extracted_data
    
    def save_dict_to_ckpt(self, submap_id: int, gaussians: GaussianModel, 
                          cameras: Dict = None) -> Optional[str]:
        """
        保存子图到磁盘
        
        Args:
            submap_id: 子图ID
            gaussians: 全局高斯模型（需要提取属于该子图的部分）
            cameras: 关键帧相机字典（可选）
            
        Returns:
            str: 保存的文件路径，如果保存失败则返回None
        """
        if self.submap_dir is None:
            Log(f"Warning: Submap directory not set (save_dir={self.save_dir}), cannot save submap {submap_id}", tag="Submap")
            return None
        
        submap = self.submaps.get(submap_id)
        if submap is None:
            Log(f"Error: Submap {submap_id} not found", tag="Submap")
            return None
        
        if not submap.keyframes:
            Log(f"Warning: Submap {submap_id} has no keyframes, skipping save", tag="Submap")
            return None
        
        try:
            # 1. 从全局高斯模型中提取属于该子图的高斯点
            gaussians_data = self._extract_submap_gaussians(gaussians, submap.keyframes)
            if gaussians_data is None:
                return None
            
            # 2. 构建保存字典
            checkpoint_dict = {
                'submap_id': submap_id,
                'start_frame_idx': submap.start_frame_idx,
                'end_frame_idx': submap.end_frame_idx,
                'keyframes': submap.keyframes,
                'gaussians': gaussians_data,
            }
            
            # 3. 可选：保存关键帧位姿
            if self.save_keyframe_poses and cameras is not None:
                keyframe_poses = {}
                for kf_id in submap.keyframes:
                    if kf_id in cameras:
                        cam = cameras[kf_id]
                        keyframe_poses[kf_id] = {
                            'R': cam.R.detach().cpu().clone() if hasattr(cam, 'R') else None,
                            'T': cam.T.detach().cpu().clone() if hasattr(cam, 'T') else None,
                            'R_gt': cam.R_gt.detach().cpu().clone() if hasattr(cam, 'R_gt') else None,
                            'T_gt': cam.T_gt.detach().cpu().clone() if hasattr(cam, 'T_gt') else None,
                        }
                checkpoint_dict['keyframe_poses'] = keyframe_poses
            
            # 4. 确定保存路径
            ckpt_filename = f"submap_{submap_id:04d}.ckpt"
            ckpt_path = os.path.join(self.submap_dir, ckpt_filename)
            
            # 5. 保存到磁盘
            torch.save(checkpoint_dict, ckpt_path)
            
            # 6. 更新子图信息
            submap.ckpt_path = ckpt_path
            submap.is_saved = True
            
            # 计算文件大小
            file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
            Log(f"Saved submap {submap_id} to {ckpt_path} ({file_size_mb:.2f} MB)", tag="Submap")
            
            return ckpt_path
            
        except Exception as e:
            Log(f"Error saving submap {submap_id}: {e}", tag="Submap")
            import traceback
            traceback.print_exc()
            return None
    
    def get_current_submap(self) -> Optional[Submap]:
        """获取当前活跃子图"""
        if self.current_submap_id is not None:
            return self.submaps[self.current_submap_id]
        return None
    
    def get_submap(self, submap_id: int) -> Optional[Submap]:
        """根据ID获取子图"""
        return self.submaps.get(submap_id)
    
    def merge_submaps_to_ply(self, output_path: str = None) -> Optional[str]:
        """
        加载所有保存的子图，拼接在一起，保存为PLY文件
        
        Args:
            output_path: 输出PLY文件路径，如果为None则使用默认路径
            
        Returns:
            str: 保存的文件路径，如果失败则返回None
        """
        if not self.enabled or self.submap_dir is None:
            Log("Submap feature is disabled or submap_dir not set, cannot merge submaps", tag="Submap")
            return None
        
        # 获取所有已保存的子图文件
        import glob
        import time
        
        # 先获取所有子图ID（包括已保存和未保存的）
        all_submap_ids = sorted(self.submaps.keys())
        Log(f"Total submaps (including unsaved): {len(all_submap_ids)}", tag="Submap")
        
        # 获取所有已保存的ckpt文件（多次尝试，确保包含最新保存的文件）
        ckpt_files = []
        max_attempts = 3
        for attempt in range(max_attempts):
            ckpt_files = sorted(glob.glob(os.path.join(self.submap_dir, "submap_*.ckpt")))
            if len(ckpt_files) >= len(all_submap_ids):
                break
            if attempt < max_attempts - 1:
                time.sleep(0.1)  # 等待100ms后重试
        
        # 检查是否有未保存的子图
        saved_submap_ids = set()
        for ckpt_file in ckpt_files:
            try:
                checkpoint = torch.load(ckpt_file, map_location='cpu')
                saved_submap_ids.add(checkpoint.get('submap_id', -1))
            except:
                pass
        
        unsaved_submaps = [sid for sid in all_submap_ids if sid not in saved_submap_ids]
        if unsaved_submaps:
            Log(f"Warning: Found {len(unsaved_submaps)} unsaved submaps: {unsaved_submaps}. They will be skipped in merge.", tag="Submap")
        
        if len(ckpt_files) == 0:
            Log("No submap checkpoint files found", tag="Submap")
            return None
        
        Log(f"Found {len(ckpt_files)} submap checkpoint files (expected {len(all_submap_ids)}), starting merge...", tag="Submap")
        
        # 收集所有子图的3D点
        all_xyz = []
        all_features_dc = []
        all_features_rest = []
        all_opacity = []
        all_scaling = []
        all_rotation = []
        
        for ckpt_file in ckpt_files:
            try:
                checkpoint = torch.load(ckpt_file, map_location='cpu')
                submap_id = checkpoint.get('submap_id', -1)
                gaussians_data = checkpoint.get('gaussians', {})
                
                if gaussians_data is None or len(gaussians_data) == 0:
                    Log(f"Warning: Submap {submap_id} has no gaussians data, skipping", tag="Submap")
                    continue
                
                # 提取3D点数据
                xyz = gaussians_data.get('_xyz')
                features_dc = gaussians_data.get('_features_dc')
                features_rest = gaussians_data.get('_features_rest')
                opacity = gaussians_data.get('_opacity')
                scaling = gaussians_data.get('_scaling')
                rotation = gaussians_data.get('_rotation')
                
                if xyz is not None:
                    all_xyz.append(xyz)
                    all_features_dc.append(features_dc)
                    all_features_rest.append(features_rest)
                    all_opacity.append(opacity)
                    all_scaling.append(scaling)
                    all_rotation.append(rotation)
                    
                    num_points = xyz.shape[0]
                    Log(f"Loaded submap {submap_id}: {num_points} points from {ckpt_file}", tag="Submap")
                else:
                    Log(f"Warning: Submap {submap_id} has no xyz data, skipping", tag="Submap")
                    
            except Exception as e:
                Log(f"Error loading submap from {ckpt_file}: {e}", tag="Submap")
                import traceback
                traceback.print_exc()
                continue
        
        if len(all_xyz) == 0:
            Log("No valid gaussians data found in any submap", tag="Submap")
            return None
        
        # 拼接所有3D点
        Log(f"Merging {len(all_xyz)} submaps...", tag="Submap")
        merged_xyz = torch.cat(all_xyz, dim=0)
        merged_features_dc = torch.cat(all_features_dc, dim=0)
        merged_features_rest = torch.cat(all_features_rest, dim=0)
        merged_opacity = torch.cat(all_opacity, dim=0)
        merged_scaling = torch.cat(all_scaling, dim=0)
        merged_rotation = torch.cat(all_rotation, dim=0)
        
        total_points = merged_xyz.shape[0]
        Log(f"Merged {total_points} points from {len(all_xyz)} submaps", tag="Submap")
        
        # 创建临时的GaussianModel对象用于保存PLY
        from gaussian_splatting.scene.gaussian_model import GaussianModel
        model_params = self.config.get("model_params", {})
        sh_degree = model_params.get("sh_degree", 0)
        merged_gaussians = GaussianModel(sh_degree, config=self.config)
        
        # 设置拼接后的数据（确保在CPU上，因为save_ply需要CPU数据）
        merged_gaussians._xyz = merged_xyz.cpu()
        merged_gaussians._features_dc = merged_features_dc.cpu()
        merged_gaussians._features_rest = merged_features_rest.cpu()
        merged_gaussians._opacity = merged_opacity.cpu()
        merged_gaussians._scaling = merged_scaling.cpu()
        merged_gaussians._rotation = merged_rotation.cpu()
        merged_gaussians.active_sh_degree = sh_degree
        merged_gaussians.max_sh_degree = sh_degree
        
        # 设置输出路径
        if output_path is None:
            output_path = os.path.join(self.save_dir, "merged_submaps.ply")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # 保存为PLY文件
        try:
            merged_gaussians.save_ply(output_path)
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            Log(f"Successfully saved merged submaps to {output_path} ({file_size_mb:.2f} MB, {total_points} points)", tag="Submap")
            return output_path
        except Exception as e:
            Log(f"Error saving merged submaps to {output_path}: {e}", tag="Submap")
            import traceback
            traceback.print_exc()
            return None

