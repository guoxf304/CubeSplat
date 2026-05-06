#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import open3d as o3d
import numpy as np
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh


def render(
    viewpoint_camera, # 相机
    pc: GaussianModel, # 高斯点云
    pipe, # 配置（如是否使用 SHs、是否预计算协方差）
    bg_color: torch.Tensor, # 背景颜色
    scaling_modifier=1.0, # 缩放因子
    override_color=None,
    mask=None,
    face_key='front',
    use_quantized=None,
):
    """
    Render the scene.渲染整个场景，将高斯点云从当前相机视角投影并生成图像、深度图等结果。

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if pc.get_xyz.shape[0] == 0: # # 若点云为空，则返回 None
        return None

    screenspace_points = ( # 创建一个和点数相同的、可求导的屏幕空间点数组（用于计算梯度）
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Set up rasterization configuration
    # 计算相机视角的水平和垂直视场角（tangent，用于投影）
    viewmatrix = viewpoint_camera.get_face_view_transform(face_key) # 61%
    projmatrix = viewpoint_camera.get_face_proj_transform(face_key) # 20%
    projmatrix_raw = viewpoint_camera.get_face_proj_raw_transform(face_key) # 0%
    campos = viewpoint_camera.get_face_camera_center(face_key) # 21%
    cam_rot_delta = viewpoint_camera.cam_rot_delta
    cam_trans_delta = viewpoint_camera.cam_trans_delta
    '''
    # 调试代码使用
    viewmatrix_front = viewpoint_camera.get_face_view_transform('front')
    C2W_front = torch.linalg.inv(viewmatrix_front.transpose(0, 1)).cpu().numpy()
    viewmatrix_back = viewpoint_camera.get_face_view_transform('back')
    C2W_back = torch.linalg.inv(viewmatrix_back.transpose(0, 1)).cpu().numpy()
    viewmatrix_left = viewpoint_camera.get_face_view_transform('left')
    C2W_left = torch.linalg.inv(viewmatrix_left.transpose(0, 1)).cpu().numpy()
    viewmatrix_right = viewpoint_camera.get_face_view_transform('right')
    C2W_right = torch.linalg.inv(viewmatrix_right.transpose(0, 1)).cpu().numpy()
    C2W_list = [C2W_front, C2W_left, C2W_right, C2W_back]
    C2W = torch.linalg.inv(viewmatrix.transpose(0, 1)).cpu().numpy()
    pcd = visualize_gaussian_point_cloud(pc)
    geometries = draw_camera_frustums(C2W_list, scale=2)
    geometries.append(pcd)
'''
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5) # 投影参数
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    relative_rotations = viewpoint_camera.get_cubemap_relative_rotation()

    rel_R = relative_rotations[face_key].to(viewmatrix)

    # 构建光栅化器配置（包括相机参数和场景设置）
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height), # 图像高
        image_width=int(viewpoint_camera.image_width), # 图像宽
        tanfovx=tanfovx, # 水平视角
        tanfovy=tanfovy, # 垂直视角
        bg=bg_color, # 背景颜色
        scale_modifier=scaling_modifier, # 缩放因子
        viewmatrix=viewmatrix, # 相机视图矩阵
        projmatrix=projmatrix, # 投影矩阵
        projmatrix_raw=projmatrix_raw, # 原始投影矩阵（无归一化）
        sh_degree=pc.active_sh_degree, # 当前的SH阶数
        campos=campos, # 相机位置
        prefiltered=False,
        debug=False,
        rel_R = rel_R,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings) # 构建光栅化器实例

    # 可视化：o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(means3D.detach().cpu().numpy())), o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)])
    
    # 根据标志/量化状态选择参数
    if use_quantized is True:
        means3D = (
            pc.get_quantized_xyz(requires_grad=True)
            if pc.quantization_enabled and pc.quantization_centers['xyz'] is not None
            else pc.get_xyz
        )
    elif use_quantized is False:
        means3D = pc.get_xyz
    else:
        if pc.quantization_enabled and pc.quantization_centers['xyz'] is not None:
            means3D = pc.get_quantized_xyz(requires_grad=True)
        else:
            means3D = pc.get_xyz
    
    means2D = screenspace_points # 高斯的屏幕空间位置（可导）
    opacity = pc.get_opacity # 不透明度（用于 alpha blending）

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    # 获取 3D 协方差（是否从 Python 中显式计算）
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier) # 显式协方差
    else:
        # 根据标志/量化状态选择缩放与旋转
        if use_quantized is True:
            scaling_param = (
                pc.get_quantized_scaling(requires_grad=True)
                if pc.quantization_enabled and pc.quantization_centers['scale'] is not None
                else pc.get_scaling
            )
            rotation_param = (
                pc.get_quantized_rotation(requires_grad=True)
                if pc.quantization_enabled and pc.quantization_centers['rot'] is not None
                else pc.get_rotation
            )
        elif use_quantized is False:
            scaling_param = pc.get_scaling
            rotation_param = pc.get_rotation
        else:
            if pc.quantization_enabled and pc.quantization_centers['scale'] is not None:
                scaling_param = pc.get_quantized_scaling(requires_grad=True)
            else:
                scaling_param = pc.get_scaling
            if pc.quantization_enabled and pc.quantization_centers['rot'] is not None:
                rotation_param = pc.get_quantized_rotation(requires_grad=True)
            else:
                rotation_param = pc.get_rotation
        
        # check if the covariance is isotropic
        # 若为各向同性（1个通道），则复制到3个通道
        if scaling_param.shape[-1] == 1:
            scales = scaling_param.repeat(1, 3)
        else:
            scales = scaling_param
        rotations = rotation_param

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 准备颜色信息（如果未预先提供，就基于球谐系数计算颜色）
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python: # 在 Python 中计算 SH → RGB
            # 使用pc.get_features（内部已处理量化开关）
            features = pc.get_features
            
            shs_view = features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = means3D - viewpoint_camera.camera_center.repeat(
                features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # 使用pc.get_features
            shs = pc.get_features
    else:
        colors_precomp = override_color # 若传入 override_color，直接使用

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # 核心渲染，光栅化阶段，调用 rasterizer 执行光栅化
    if mask is not None: # 如果传入 mask，只渲染部分高斯
        rendered_image, radii, depth, opacity = rasterizer(
            means3D=means3D[mask],
            means2D=means2D[mask],
            shs=shs[mask],
            colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
            opacities=opacity[mask],
            scales=scales[mask],
            rotations=rotations[mask],
            cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
            theta=cam_rot_delta, # 位姿增量旋转（用于优化
            rho=cam_trans_delta, # 位姿增量平移（用于优化）
        )
    else: # 默认情况下渲染全部高斯
        rendered_image, radii, depth, opacity, n_touched = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            theta=cam_rot_delta,
            rho=cam_trans_delta,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image, # 渲染后的图像
        "viewspace_points": screenspace_points, # 用于求导的屏幕空间高斯位置 [N, 2]
        "visibility_filter": radii > 0, # 过滤不可见（半径为 0）的高斯，bool [N,]
        "radii": radii, # 高斯在屏幕上的半径（表示影响范围） [N,]
        "depth": depth, # 渲染得到的深度图 [1, H, W]
        "opacity": opacity, # 渲染得到的不透明度图 [1, H, W]
        "n_touched": n_touched, # 每个高斯点命中的像素个数。int [N,]
    }

'''
def draw_camera_frustums(c2w_list, scale=0.1, color=[0.0, 0.0, 1.0]):
    """
    使用 Open3D 可视化多个相机视锥
    :param c2w_list: list of 4x4 numpy arrays (相机的 c2w 位姿)
    :param scale: 视锥尺寸
    :param color: 颜色（RGB 0~1）
    :return: open3d.geometry.Geometry objects 可直接用 o3d.visualization.draw_geometries 显示
    """
    geometries = []

    # 相机坐标系下的 frustum 顶点（小金字塔）
    frustum = np.array([
        [0, 0, 0],  # 相机中心
        [-1, -1, 1],  # 左下
        [1, -1, 1],  # 右下
        [1, 1, 1],  # 右上
        [-1, 1, 1]  # 左上
    ]).T * scale  # (3, 5)

    frustum_hom = np.vstack((frustum, np.ones((1, frustum.shape[1]))))  # (4, 5)

    for c2w in c2w_list:
        pts_w = (c2w @ frustum_hom)[:3, :]  # 世界坐标下的5个点
        points = [pts_w[:, i] for i in range(5)]

        # 创建线段连接视锥边
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # 从中心到四角
            [1, 2], [2, 3], [3, 4], [4, 1]  # 外框四边
        ]
        colors = [color for _ in lines]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)

        geometries.append(line_set)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(coordinate_frame)
    return geometries

def visualize_gaussian_point_cloud(pc):
    means3D = pc.get_xyz.detach().cpu().numpy()  # 获取高斯点的世界坐标
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means3D)
    return pcd

'''