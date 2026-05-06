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

import os

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from bitarray import bitarray
from collections import OrderedDict
# from utils.debug_function import draw_camera_frustums

from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View2, getWorld2View2face
from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.system_utils import mkdir_p


class GaussianModel:
    # sh_degree球谐函数的阶数
    def __init__(self, sh_degree: int, config=None):
        self.active_sh_degree = 0                               # 球谐函数阶数
        self.max_sh_degree = sh_degree                          # 最大阶数

        self._xyz = torch.empty(0, device="cuda")               # 3D 坐标 (xyz),torch.Size([N, 3])
        self._features_dc = torch.empty(0, device="cuda")       # 直流分量（0阶）的特征，torch.Size([N, 1, 3])
        self._features_rest = torch.empty(0, device="cuda")     # 剩余分量（高阶），torch.Size([N, 15, 3])
        self._scaling = torch.empty(0, device="cuda")           # 高斯球大小或者缩放，torch.Size([N, 3])
        self._rotation = torch.empty(0, device="cuda")          # 高斯球旋转信息，torch.Size([N, 4])
        self._opacity = torch.empty(0, device="cuda")           # 透明度，torch.Size([N, 1])
        self.max_radii2D = torch.empty(0, device="cuda")        # 最大半径，torch.Size([N])
        self.xyz_gradient_accum = torch.empty(0, device="cuda") # 位置梯度累积，torch.Size([N, 1])

        self.unique_kfIDs = torch.empty(0).int()                # 关键帧ID，torch.Size([N])
        self.n_obs = torch.empty(0).int()                       # 被观测的次数，torch.Size([N])

        self.optimizer = None                                   # 优化器

        self.scaling_activation = torch.exp                     # 定义的缩放函数
        self.scaling_inverse_activation = torch.log             # 定义的缩放的逆激活函数

        self.covariance_activation = self.build_covariance_from_scaling_rotation # 定义协方差的激活方式

        self.opacity_activation = torch.sigmoid                 # 透明度激活函数
        self.inverse_opacity_activation = inverse_sigmoid       # 透明度激活函数的逆函数

        self.rotation_activation = torch.nn.functional.normalize    # 旋转激活函数

        self.config = config
        self.ply_input = None

        self.isotropic = False

        # K-means quantization related attributes
        self.kmeans_quantizers = {}  # 量化器字典
        self.kmeans_config = None    # 量化配置
        self.quantization_enabled = False
        
        # 聚类中心和索引存储（不带梯度）
        self.quantization_centers = {
            'xyz': None,
            'dc': None, 
            'sh': None,
            'scale': None,
            'rot': None,
        }
        self.quantization_indices = {
            'xyz': None,
            'dc': None,
            'sh': None, 
            'scale': None,
            'rot': None,
        }
        
        # 初始化量化器
        self.init_quantizers(config['Training']['kmeans_params'])
        

    def init_quantizers(self, kmeans_config):
        """初始化k-means量化器"""
        from gaussian_splatting.scene.kmeans_quantize import Quantize_kMeans
        
        self.kmeans_config = kmeans_config
        self.quantization_enabled = kmeans_config.get('enabled', False)
        
        if not self.quantization_enabled:
            return
        
        # 为每种参数类型创建量化器
        quantized_params = kmeans_config.get('quantized_params', [])
        
        for param_type in quantized_params:
            if param_type == 'dc':
                n_clusters = kmeans_config.get('n_clusters_dc', 4096)
            elif param_type == 'sh':
                n_clusters = kmeans_config.get('n_clusters_sh', 512)
            else:
                n_clusters = kmeans_config.get('n_clusters', 4096)
            
            n_iters = kmeans_config.get('kmeans_iters', 10)
            
            # 创建量化器
            self.kmeans_quantizers[param_type] = Quantize_kMeans(
                num_clusters=n_clusters,
                num_iters=n_iters
            )
        
        print(f"初始化量化器完成: {list(self.kmeans_quantizers.keys())}")

    # 从尺度以及旋转构建协方差
    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        scaling = self.get_quantized_scaling(requires_grad=True)
        return self.scaling_activation(scaling)

    @property
    def get_scaling_nq(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        rotation = self.get_quantized_rotation(requires_grad=True)
        return self.rotation_activation(rotation)

    @property
    def get_xyz(self):
        xyz = self.get_quantized_xyz(requires_grad=True)
        return xyz

    @property
    def get_features(self):
        features_dc = self.get_quantized_features_dc(requires_grad=True)
        features_rest = self.get_quantized_features_rest(requires_grad=True)
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # RGB生成点云
    def create_pcd_from_Cubeimage(self, cam_info, init=False, scale=2.0, depthmap_dict=None):
        '''第二层嵌套，是extend_from_pcd_seq的子函数'''
        rgb_raw_dict = {}
        cam = cam_info
        for key, img in cam.Cubemap_image.items():
            image_ab = (torch.exp(cam.exposure_a)) * img + cam.exposure_b
            image_ab = torch.clamp(image_ab, 0.0, 1.0)
            rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            rgb_raw_dict[key] = rgb_raw

        if depthmap_dict is not None and len(depthmap_dict) > 0:
            depth_dict = {key: o3d.geometry.Image(depthmap_dict[key].astype(np.float32)) for key in cam.Cubemap_image}
            rgb_dict = {key: o3d.geometry.Image(rgb_raw_dict[key].astype(np.uint8)) for key in cam.Cubemap_image}
        else:
            # 处理每个面的深度
            depth_dict = {}
            for key in cam.Cubemap_image:
                depth_raw = cam.depth.get(key, None)  # 对应每个方向的深度
                if depth_raw is None:
                    depth_raw = np.empty((cam.image_height, cam.image_width))

                if self.config["Dataset"]["sensor_type"] == "monocular":
                    depth_raw = (
                        np.ones_like(depth_raw)
                        + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                        * 0.05
                    ) * scale
                depth_dict[key] = o3d.geometry.Image(depth_raw.astype(np.float32))
            rgb_dict = {key: o3d.geometry.Image(rgb_raw_dict[key].astype(np.uint8)) for key in cam.Cubemap_image}

        # 现在我们有了每个方向的RGB图和深度图，调用 create_pcd_from_image_and_depth
        pcd_dict = {}
        for key in cam.Cubemap_image:
            rgb = rgb_dict[key]
            depth = depth_dict[key]
            fused_point_cloud, features, scales, rots, opacities = self.create_pcd_from_image_and_depth(
                cam, rgb, depth, init, face_key=key
            )
            # 保存每面对应的 PCD 数据
            pcd_dict[key] = {
                "points": fused_point_cloud,
                "features": features,
                "scales": scales,
                "rots": rots,
                "opacities": opacities
            }

        return pcd_dict

    # RGB+depth 生成点云
    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False, face_key = 'front'):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"] # 32
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"] # 0.01
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                # 将Open3D Image转换为numpy数组，并只考虑非零值
                if isinstance(depth, o3d.geometry.Image):
                    depth_np = np.asarray(depth)
                else:
                    depth_np = depth
                # 只考虑非零深度值计算中位数（mask外的区域为0，不参与计算）
                valid_depths = depth_np[depth_np > 0]
                if len(valid_depths) > 0:
                    median_depth = np.median(valid_depths)
                    point_size = min(0.05, point_size * median_depth)
                # 如果没有有效深度值，使用默认point_size
        # 检查深度图是否有效（至少有一些非零值）
        if isinstance(depth, o3d.geometry.Image):
            depth_np = np.asarray(depth)
        else:
            depth_np = depth
        valid_depth_count = np.sum(depth_np > 0)
        
        # 如果深度图全为0或几乎没有有效值，使用随机深度（避免Open3D操作失败）
        if valid_depth_count < 10:  # 至少需要10个有效像素
            # 生成随机深度作为fallback
            depth_np = (
                np.ones((cam.image_height, cam.image_width))
                + (np.random.randn(cam.image_height, cam.image_width) - 0.5) * 0.05
            ) * 2.0  # 使用默认scale 2.0
            depth = o3d.geometry.Image(depth_np.astype(np.float32))
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth( # 512*512 1 channels
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )
        relative_rotations = cam.get_cubemap_relative_rotation()
        rel_R = relative_rotations[face_key].to(dtype=cam.R.dtype, device=cam.R.device)

        #new_R = rel_R @ cam.R  # 主相机旋转乘以每个面相对旋转
        #T_center = -cam.R.transpose(0, 1)  @ cam.T
        #new_T = - torch.linalg.inv(new_R.transpose(0, 1)) @ T_center

        W2C = getWorld2View2face(rel_R @ cam.R , rel_R @ cam.T ).cpu().numpy()

        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        ) # 展示：o3d.visualization.draw_geometries([pcd_tmp])
        
        # 检查点云是否为空
        if len(pcd_tmp.points) == 0:
            # 如果点云为空，创建一些默认点
            num_points = 100
            new_xyz = np.random.randn(num_points, 3).astype(np.float32) * 0.1
            new_rgb = np.ones((num_points, 3), dtype=np.float32) * 0.5
        else:
            C2W = np.linalg.inv(W2C)
            # geometries = draw_camera_frustums([C2W],scale=1)
            # geometries.append(pcd_tmp)
            pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor) # 随机采样数量缩减
            new_xyz = np.asarray(pcd_tmp.points)
            new_rgb = np.asarray(pcd_tmp.colors)

        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda()) # DC 分量
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color # DC 分量
        features[:, 3:, 1:] = 0.0 # 高阶分量

        dist2 = ( # torch.Size([N])
            torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )
        # 返回：每个点的 3D 坐标、球谐函数特征（颜色）、高斯尺度、旋转（四元数）、透明度
        return fused_point_cloud, features, scales, rots, opacities

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    # 将新的点云数据添加到当前高斯点的集合中
    def extend_from_pcd(
        self, pcd_dict, kf_id
    ):
        all_points = []
        all_features = []
        all_scales = []
        all_rots = []
        all_opacities = []

        for key, data in pcd_dict.items():
            all_points.append(data["points"])
            all_features.append(data["features"])
            all_scales.append(data["scales"])
            all_rots.append(data["rots"])
            all_opacities.append(data["opacities"])

        fused_point_cloud = torch.cat(all_points, dim=0)
        features = torch.cat(all_features, dim=0)
        scales = torch.cat(all_scales, dim=0)
        rots = torch.cat(all_rots, dim=0)
        opacities = torch.cat(all_opacities, dim=0)

        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True)) # 可训练的xyz
        new_features_dc = nn.Parameter( # 可训练的DC
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter( # 可训练高阶分量
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id # 每个新点ID都是当前帧
        new_n_obs = torch.zeros((new_xyz.shape[0])).int() # 观测次数设为0
        self.densification_postfix( # 完成拼接
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
        )

    # 更新点云模型
    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap_dict=None
    ):
        '''初始化的第一层嵌套，更新点云模型'''
        pcd_dict = (
            self.create_pcd_from_Cubeimage(cam_info, init, scale=scale, depthmap_dict=depthmap_dict)
        )
        self.extend_from_pcd(
            pcd_dict, kf_id
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # lr = self.xyz_scheduler_args(iteration)
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )

                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self, save_att=None):
        l = []
        if save_att is None:
            # Default behavior - include all attributes
            l = ["x", "y", "z", "nx", "ny", "nz"]
            # All channels except the 3 DC
            for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
                l.append("f_dc_{}".format(i))
            for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
                l.append("f_rest_{}".format(i))
            l.append("opacity")
            for i in range(self._scaling.shape[1]):
                l.append("scale_{}".format(i))
            for i in range(self._rotation.shape[1]):
                l.append("rot_{}".format(i))
            return l
        else:
            # Selective saving based on save_att
            if 'xyz' in save_att:
                l += ['x', 'y', 'z']
            if 'normals' in save_att:
                l += ['nx', 'ny', 'nz']
            # All channels except the 3 DC
            if 'f_dc' in save_att:
                for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
                    l.append('f_dc_{}'.format(i))
            if 'f_rest' in save_att:
                for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
                    l.append('f_rest_{}'.format(i))
            l.append('opacity')
            if 'scale' in save_att:
                for i in range(self._scaling.shape[1]):
                    l.append('scale_{}'.format(i))
            if 'rotation' in save_att:
                for i in range(self._rotation.shape[1]):
                    l.append('rot_{}'.format(i))
            return l

    def save_ply(self, path, save_q=[], save_attributes=None):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # Use quantized data if specified
        if 'pos' in save_q:
            xyz = self.get_quantized_xyz(requires_grad=False).detach().cpu().numpy()
        if 'dc' in save_q or 'sh_dc' in save_q:
            f_dc = self.get_quantized_features_dc(requires_grad=False).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        if 'sh' in save_q or 'sh_dc' in save_q:
            f_rest = self.get_quantized_features_rest(requires_grad=False).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        if 'scale' in save_q or 'scale_rot' in save_q:
            scale = self.get_quantized_scaling(requires_grad=False).detach().cpu().numpy()
        if 'rot' in save_q or 'scale_rot' in save_q:
            rotation = self.get_quantized_rotation(requires_grad=False).detach().cpu().numpy()

        all_attributes = {'xyz': xyz, 'f_dc': f_dc, 'f_rest': f_rest, 'opacities': opacities,
                          'scale': scale, 'rotation': rotation}
        if save_attributes is None:
            save_attributes = list(all_attributes.keys())

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes(save_attributes)
        ]
        print('non-quantized attributes: ', save_attributes)
        print('quantized attributes: ', save_q)

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(tuple([val for (key, val) in all_attributes.items() if key in save_attributes]), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters
    ):  ##Reset opacity for only non-visible gaussians
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def bin2dec(self, b, bits):
        """Convert binary b to decimal integer.

        Code from: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
        """
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, torch.int64)
        return torch.sum(mask * b, -1)

    def load_ply(self, path, load_quant=False):
        plydata = PlyData.read(path)
        quant_params = []
        # Load indices and codebook for quantized params
        if load_quant:
            base_path = '/'.join(path.split('/')[:-1])
            inds_file = os.path.join(base_path, 'kmeans_inds.bin')
            codebook_file = os.path.join(base_path, 'kmeans_centers.pth')
            args_file = os.path.join(base_path, 'kmeans_args.npy')
            codebook = torch.load(codebook_file)
            args_dict = np.load(args_file, allow_pickle=True).item()
            quant_params = args_dict['params']
            loaded_bitarray = bitarray()
            with open(inds_file, 'rb') as file:
                loaded_bitarray.fromfile(file)
            # bitarray pads 0s if array is not divisible by 8. ignore extra 0s at end when loading
            total_len = args_dict['total_len']
            loaded_bitarray = loaded_bitarray[:total_len].tolist()
            indices = np.reshape(loaded_bitarray, (-1, args_dict['n_bits']))
            indices = self.bin2dec(torch.from_numpy(indices), args_dict['n_bits'])
            indices = np.reshape(indices.cpu().numpy(), (len(quant_params), -1))
            indices_dict = OrderedDict()
            for i, key in enumerate(args_dict['params']):
                indices_dict[key] = indices[i]

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            # 3DGS 标准 .ply 可能没有法线 (nx, ny, nz)，无则用零向量
            try:
                normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            except (ValueError, KeyError):
                normals = np.zeros_like(positions)
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        
        if 'xyz' in quant_params:
            xyz = np.expand_dims(codebook['xyz'][indices_dict['xyz']].cpu().numpy(), -1)
        else:
            xyz = np.stack(
                (
                    np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"]),
                ),
                axis=1,
            )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        if 'dc' in quant_params:
            features_dc = np.expand_dims(codebook['dc'][indices_dict['dc']].cpu().numpy(), -1)
        else:
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if 'sh' in quant_params:
            features_extra = codebook['sh'][indices_dict['sh']].cpu().numpy()
            features_extra = features_extra.reshape((features_extra.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3))
            features_extra = features_extra.transpose((0, 2, 1))
        else:
            extra_f_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith("f_rest_")
            ]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            expected_rest = 3 * (self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # 兼容不同来源的 .ply（如 PhotoSLAM/MonoGS）：数量不足补零，过多截断
            if features_extra.shape[1] != expected_rest:
                buf = np.zeros((xyz.shape[0], expected_rest), dtype=features_extra.dtype)
                n = min(features_extra.shape[1], expected_rest)
                buf[:, :n] = features_extra[:, :n]
                features_extra = buf
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        if 'scale' in quant_params:
            scales = codebook['scale'][indices_dict['scale']].cpu().numpy()
        else:
            scale_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith("scale_")
            ]
            scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if 'rot' in quant_params:
            rots = codebook['rot'][indices_dict['rot']].cpu().numpy()
        else:
            rot_names = [
                p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
            ]
            rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
            rots = np.zeros((xyz.shape[0], len(rot_names)))
            for idx, attr_name in enumerate(rot_names):
                rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.unique_kfIDs = torch.zeros((self._xyz.shape[0]))
        self.n_obs = torch.zeros((self._xyz.shape[0]), device="cpu").int()

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        '''清除信息，在下面的函数prune_points中使用到'''
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        '''清除所有的高斯球，在后端初始化中使用到'''
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]
        
        # 量化状态失效：后续会整模型重新量化，这里直接清空
        for k in self.quantization_centers.keys():
            self.quantization_centers[k] = None
            self.quantization_indices[k] = None

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
    ):
        '''完成高斯点云的拼接'''
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d) # 将训练数据添加到原来的训练参数中
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 量化状态失效：后续会整模型重新量化，这里直接清空
        for k in self.quantization_centers.keys():
            self.quantization_centers[k] = None
            self.quantization_indices[k] = None

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()

    def _sync_quantization_indices_after_densification(self, new_points_count):
        """为新添加的高斯点同步量化索引"""
        if not self.quantization_enabled or not self.quantization_indices:
            return
        
        # 计算原始高斯点数量
        original_count = self.quantization_indices['dc'].shape[0] if self.quantization_indices['dc'] is not None else 0
        
        # 为新添加的高斯点分配量化索引
        for param_type in self.quantization_indices:
            if self.quantization_indices[param_type] is not None:
                # 获取新添加的高斯点参数
                new_points_params = self._get_new_points_params(param_type, original_count, new_points_count)
                
                if new_points_params is not None:
                    # 找到最近的聚类中心
                    new_indices = self._find_nearest_cluster_centers(new_points_params, param_type)
                    
                    if new_indices is not None:
                        # 拼接原始索引和新索引
                        self.quantization_indices[param_type] = torch.cat([
                            self.quantization_indices[param_type], 
                            new_indices
                        ])
                        print(f"量化索引同步 ({param_type}): 为 {new_points_count} 个新点分配了最近聚类索引")
                    else:
                        # 如果寻找最近中心失败，使用随机分配
                        num_centers = self.quantization_centers[param_type].shape[0]
                        new_indices = torch.randint(0, num_centers, (new_points_count,), 
                                                  device=self.quantization_indices[param_type].device)
                        self.quantization_indices[param_type] = torch.cat([
                            self.quantization_indices[param_type], 
                            new_indices
                        ])
                        print(f"量化索引同步 ({param_type}): 使用随机分配作为后备方案")
                else:
                    # 如果无法获取参数，使用随机分配作为后备
                    num_centers = self.quantization_centers[param_type].shape[0]
                    new_indices = torch.randint(0, num_centers, (new_points_count,), 
                                              device=self.quantization_indices[param_type].device)
                    self.quantization_indices[param_type] = torch.cat([
                        self.quantization_indices[param_type], 
                        new_indices
                    ])

    def _get_new_points_params(self, param_type, original_count, new_points_count):
        """获取新添加的高斯点参数"""
        try:
            # 映射参数类型到实际属性名
            attr_mapping = {
                'dc': '_features_dc',
                'sh': '_features_rest', 
                'scale': '_scaling',
                'rot': '_rotation',
                'xyz': '_xyz'
            }
            
            if param_type not in attr_mapping:
                return None
                
            attr_name = attr_mapping[param_type]
            param_tensor = getattr(self, attr_name)
            
            # 获取新添加的参数（从原始数量开始）
            if param_tensor.shape[0] >= original_count + new_points_count:
                new_params = param_tensor[original_count:original_count + new_points_count]
                return new_params
            else:
                return None
                
        except Exception as e:
            print(f"获取新点参数失败 ({param_type}): {e}")
            return None

    def _find_nearest_cluster_centers(self, new_params, param_type):
        """为新参数找到最近的聚类中心"""
        try:
            centers = self.quantization_centers[param_type]
            
            if centers is None or new_params is None:
                print(f"聚类中心或新参数为空 ({param_type})")
                return None
            
            # 确保参数形状正确
            original_shape = new_params.shape
            if len(new_params.shape) == 3:  # [N, 1, C] -> [N, C]
                new_params = new_params.squeeze(1)
            elif len(new_params.shape) == 2 and new_params.shape[1] == 1:  # [N, 1] -> [N]
                new_params = new_params.squeeze(1)
            
            # 检查形状兼容性
            if len(new_params.shape) != 2 or len(centers.shape) != 2:
                print(f"参数形状不兼容: new_params {new_params.shape}, centers {centers.shape}")
                return None
            
            if new_params.shape[1] != centers.shape[1]:
                print(f"特征维度不匹配: new_params {new_params.shape[1]}, centers {centers.shape[1]}")
                return None
            
            # 计算距离矩阵
            # new_params: [N, D], centers: [K, D]
            distances = torch.cdist(new_params, centers, p=2)  # 欧几里得距离
            
            # 找到最近的聚类中心索引
            nearest_indices = torch.argmin(distances, dim=1)
            
            # 输出一些统计信息
            min_distances = torch.min(distances, dim=1)[0]
            avg_min_distance = min_distances.mean().item()
            print(f"最近聚类分配 ({param_type}): 平均最小距离 {avg_min_distance:.4f}, 形状 {original_shape} -> {new_params.shape}")
            
            return nearest_indices
            
        except Exception as e:
            print(f"寻找最近聚类中心失败 ({param_type}): {e}")
            # 返回随机索引作为后备
            if new_params is not None:
                num_centers = self.quantization_centers[param_type].shape[0]
                return torch.randint(0, num_centers, (new_params.shape[0],), 
                                   device=new_params.device)
            return None

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling_nq, dim=1).values
            > self.percent_dense * scene_extent,
        )

        # 统计分裂的高斯点数量
        # split_count = selected_pts_mask.sum().item()
        # if split_count > 0:
        #     print(f"高斯分裂: 选择了 {split_count} 个高斯点进行分裂")

        stds = self.get_scaling_nq[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling_nq[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        # 记录分裂前的高斯点数量
        points_before = self.get_xyz.shape[0]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

        # 统计分裂后的变化
        # points_after = self.get_xyz.shape[0]
        # added_points = points_after - points_before
        # if added_points > 0:
        #     print(f"高斯分裂: 添加了 {added_points} 个新高斯点 (从 {points_before} 增加到 {points_after})")

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling_nq, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        # 统计克隆的高斯点数量
        clone_count = selected_pts_mask.sum().item()
        # if clone_count > 0:
        #     print(f"高斯克隆: 选择了 {clone_count} 个高斯点进行克隆")

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        
        # 记录克隆前的高斯点数量
        points_before = self.get_xyz.shape[0]
        
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )
        
        # 统计克隆后的变化
        # points_after = self.get_xyz.shape[0]
        # added_points = points_after - points_before
        # if added_points > 0:
        #     print(f"高斯克隆: 添加了 {added_points} 个新高斯点 (从 {points_before} 增加到 {points_after})")

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # 记录操作前的高斯点数量
        # points_before_operations = self.get_xyz.shape[0]
        # print(f"高斯操作开始: 当前有 {points_before_operations} 个高斯点")

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # 记录修剪前的高斯点数量
        points_before_prune = self.get_xyz.shape[0]
        
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling_nq.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        
        # 统计要修剪的高斯点数量
        # prune_count = prune_mask.sum().item()
        # if prune_count > 0:
        #     print(f"高斯修剪: 准备修剪 {prune_count} 个高斯点")
        
        self.prune_points(prune_mask)
        
        # 统计修剪后的变化
        points_after_prune = self.get_xyz.shape[0]
        pruned_points = points_before_prune - points_after_prune
        # if pruned_points > 0:
        #     print(f"高斯修剪: 修剪了 {pruned_points} 个高斯点 (从 {points_before_prune} 减少到 {points_after_prune})")
        
        # 总结整个操作
        # total_change = points_after_prune - points_before_operations
        # print(f"高斯操作完成: 总变化 {total_change:+d} 个高斯点 (从 {points_before_operations} 到 {points_after_prune})")

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict() if self.optimizer is not None else None,
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if opt_dict is not None:
            self.optimizer.load_state_dict(opt_dict)

    def prune(self, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling_nq.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def apply_quantization(self, param_types=None, assign_flag = None):
        """应用k-means量化
        - update_centers_only=True: 仅更新聚类中心（assign=False），不更新样本的聚类索引
        - update_centers_only=False: 正常量化流程（assign=True），更新中心与索引
        """
        
        if param_types is None:
            param_types = self.kmeans_config.get('quantized_params', [])
        
        for param_type in param_types:
            if param_type not in self.kmeans_quantizers:
                continue
            
            quantizer = self.kmeans_quantizers[param_type]
            
            # 执行量化

            if param_type == 'dc':
                quantizer.forward_dc(self, assign=assign_flag)
            elif param_type == 'sh':
                quantizer.forward_frest(self, assign=assign_flag)
            elif param_type == 'scale':
                quantizer.forward_scale(self, assign=assign_flag)
            elif param_type == 'rot':
                quantizer.forward_rot(self, assign=assign_flag)
            elif param_type == 'xyz':
                quantizer.forward_pos(self, assign=assign_flag)
            
            # 存储聚类中心；仅在完整量化时更新索引
            self.quantization_centers[param_type] = quantizer.centers.detach().clone()
            if assign_flag:
                self.quantization_indices[param_type] = quantizer.cls_ids.detach().clone()
            
            # 输出量化后的孤立值数量
            num_unique_values = quantizer.centers.shape[0]
            
            # 正确的属性名称映射
            attr_mapping = {
                'dc': '_features_dc',
                'sh': '_features_rest', 
                'scale': '_scaling',
                'rot': '_rotation',
                'xyz': '_xyz'
            }
            
            # if param_type in attr_mapping:
            #     original_size = getattr(self, attr_mapping[param_type]).shape[0]
            #     print(f"  {param_type}: 原始参数数量={original_size}, 量化后孤立值数量={num_unique_values}")
            # else:
            #     print(f"  {param_type}: 量化后孤立值数量={num_unique_values} (无法获取原始参数数量)")
        
        if assign_flag:
            print(f"量化中心更新完成，参数类型: {param_types}")


    def verify_quantization_effectiveness(self):
        """验证量化效果，输出详细的量化信息"""
        if not self.quantization_enabled:
            print("量化未启用")
            return
        
        # print("\n=== 量化效果验证 ===")
        
        # 正确的属性名称映射
        attr_mapping = {
            'dc': '_features_dc',
            'sh': '_features_rest', 
            'scale': '_scaling',
            'rot': '_rotation',
            'xyz': '_xyz'
        }
        
        for param_type in self.quantization_centers:
            if self.quantization_centers[param_type] is not None:
                centers = self.quantization_centers[param_type]
                indices = self.quantization_indices[param_type]
                
                # 基本信息
                num_centers = centers.shape[0]
                num_indices = indices.shape[0]
                
                # 获取原始参数大小
                if param_type in attr_mapping:
                    original_param = getattr(self, attr_mapping[param_type])
                    original_shape = original_param.shape
                else:
                    original_shape = "N/A"
                
                # 计算压缩比
                compressed_size = centers.numel() + indices.numel()
                original_size = original_param.numel() if param_type in attr_mapping else 0
                compression_ratio = compressed_size / original_size if original_size > 0 else 0
                
                # 验证索引范围
                min_index = indices.min().item()
                max_index = indices.max().item()
                
                # print(f"{param_type}:")
                # print(f"  原始参数维度: {original_shape}")
                # print(f"  聚类中心维度: {centers.shape}")
                # print(f"  索引数量: {num_indices}")
                # print(f"  压缩比: {compression_ratio:.4f}")
                # print(f"  索引范围: [{min_index}, {max_index}]")
                # print(f"  索引是否有效: {min_index >= 0 and max_index < num_centers}")
                
                # 验证量化值的一致性
                if param_type == 'dc':
                    quantized_values = self.get_quantized_features_dc(requires_grad=False)
                elif param_type == 'sh':
                    quantized_values = self.get_quantized_features_rest(requires_grad=False)
                elif param_type == 'scale':
                    quantized_values = self.get_quantized_scaling(requires_grad=False)
                elif param_type == 'rot':
                    quantized_values = self.get_quantized_rotation(requires_grad=False)
                elif param_type == 'xyz':
                    quantized_values = self.get_quantized_xyz(requires_grad=False)
                else:
                    continue
                
                # 检查量化值是否真的来自聚类中心
                unique_quantized = torch.unique(quantized_values.view(-1), dim=0)
                # print(f"  量化后唯一值数量: {unique_quantized.shape[0]}")
                # print(f"  量化是否有效: {unique_quantized.shape[0] <= num_centers}")
                # print()

    def get_quantized_features_dc(self, requires_grad=True):
        """获取量化的DC特征"""
        if self.quantization_centers['dc'] is None:
            return self._features_dc
        
        centers = self.quantization_centers['dc']
        indices = self.quantization_indices['dc']
        
        # 检查索引与当前参数大小是否匹配
        if indices.shape[0] != self._features_dc.shape[0]:
            print(f"警告: 量化索引大小({indices.shape[0]})与参数大小({self._features_dc.shape[0]})不匹配，使用原始参数")
            return self._features_dc
        
        if requires_grad:
            # 带梯度的计算
            sampled_centers = torch.gather(centers, 0, indices.unsqueeze(-1).repeat(1, centers.shape[1]))
            # 期望形状与 _features_dc 一致 (N, 1, 3)
            if sampled_centers.dim() == 2 and sampled_centers.shape[1] == 3:
                sampled_centers = sampled_centers.unsqueeze(1)
            quantized_features_dc = self._features_dc - self._features_dc.detach() + sampled_centers
            return quantized_features_dc  # DC特征通常不需要激活函数
        else:
            # 不带梯度的计算
            sampled_centers = torch.gather(centers, 0, indices.unsqueeze(-1).repeat(1, centers.shape[1]))
            if sampled_centers.dim() == 2 and sampled_centers.shape[1] == 3:
                sampled_centers = sampled_centers.unsqueeze(1)
            return sampled_centers

    def get_quantized_features_rest(self, requires_grad=True):
        """获取量化的SH特征"""
        if self.quantization_centers['sh'] is None:
            return self._features_rest
        
        centers = self.quantization_centers['sh']
        indices = self.quantization_indices['sh']
        
        # 检查索引与当前参数大小是否匹配
        if indices.shape[0] != self._features_rest.shape[0]:
            print(f"警告: 量化索引大小({indices.shape[0]})与参数大小({self._features_rest.shape[0]})不匹配，使用原始参数")
            return self._features_rest
        
        if requires_grad:
            sampled_centers = torch.gather(centers, 0, indices.unsqueeze(-1).repeat(1, centers.shape[1]))
            # 期望形状与 _features_rest 一致 (N, C, 3)
            C = self._features_rest.shape[1]
            if sampled_centers.dim() == 2:
                if sampled_centers.shape[1] == C * 3:
                    sampled_centers = sampled_centers.view(-1, C, 3)
                elif sampled_centers.shape[1] == 3:
                    sampled_centers = sampled_centers.unsqueeze(1).expand(-1, C, -1)
            quantized_features_rest = self._features_rest - self._features_rest.detach() + sampled_centers
            return quantized_features_rest  # SH特征通常不需要激活函数
        else:
            sampled_centers = torch.gather(centers, 0, indices.unsqueeze(-1).repeat(1, centers.shape[1]))
            C = self._features_rest.shape[1]
            if sampled_centers.dim() == 2:
                if sampled_centers.shape[1] == C * 3:
                    sampled_centers = sampled_centers.view(-1, C, 3)
                elif sampled_centers.shape[1] == 3:
                    sampled_centers = sampled_centers.unsqueeze(1).expand(-1, C, -1)
            return sampled_centers

    def get_quantized_scaling(self, requires_grad=True):
        """获取量化的缩放参数"""
        if self.quantization_centers['scale'] is None:
            return self._scaling
        
        centers = self.quantization_centers['scale'] # k
        indices = self.quantization_indices['scale'] # N
        
        # 检查索引与当前参数大小是否匹配
        if indices.shape[0] != self._scaling.shape[0]:
            print(f"警告: 量化索引大小({indices.shape[0]})与参数大小({self._scaling.shape[0]})不匹配，使用原始参数")
            return self._scaling
        
        if requires_grad:
            sampled_centers = torch.gather(centers, 0, indices.unsqueeze(-1).repeat(1, centers.shape[1]))
            quantized_scaling = self._scaling - self._scaling.detach() + sampled_centers
            return self.scaling_activation(quantized_scaling)
        else:
            sampled_centers = torch.gather(centers, 0, indices.unsqueeze(-1).repeat(1, centers.shape[1]))
            return sampled_centers

    def get_quantized_rotation(self, requires_grad=True):
        """获取量化的旋转参数"""
        if self.quantization_centers['rot'] is None:
            return self._rotation
        
        centers = self.quantization_centers['rot']
        indices = self.quantization_indices['rot']
        
        # 检查索引与当前参数大小是否匹配
        if indices.shape[0] != self._rotation.shape[0]:
            print(f"警告: 量化索引大小({indices.shape[0]})与参数大小({self._rotation.shape[0]})不匹配，使用原始参数")
            return self._rotation
        
        if requires_grad:
            sampled_centers = torch.gather(centers, 0, indices.unsqueeze(-1).repeat(1, centers.shape[1]))
            quantized_rotation = self._rotation - self._rotation.detach() + sampled_centers
            return self.rotation_activation(quantized_rotation)
        else:
            sampled_centers = torch.gather(centers, 0, indices.unsqueeze(-1).repeat(1, centers.shape[1]))
            return self.rotation_activation(sampled_centers)

    def get_quantized_xyz(self, requires_grad=True):
        """获取量化的位置参数"""
        if self.quantization_centers['xyz'] is None:
            return self._xyz
        
        centers = self.quantization_centers['xyz']
        indices = self.quantization_indices['xyz']
        
        # 检查索引与当前参数大小是否匹配
        if indices.shape[0] != self._xyz.shape[0]:
            print(f"警告: 量化索引大小({indices.shape[0]})与参数大小({self._xyz.shape[0]})不匹配，使用原始参数")
            return self._xyz
        
        if requires_grad:
            sampled_centers = torch.gather(centers, 0, indices.unsqueeze(-1).repeat(1, centers.shape[1]))
            quantized_xyz = self._xyz - self._xyz.detach() + sampled_centers
            return quantized_xyz  # xyz通常不需要激活函数
        else:
            sampled_centers = torch.gather(centers, 0, indices.unsqueeze(-1).repeat(1, centers.shape[1]))
            return sampled_centers


# # 调试代码使用
# def draw_camera_frustums(c2w_list, scale=0.1, color=[0.0, 0.0, 1.0]):
#     """
#     使用 Open3D 可视化多个相机视锥
#     :param c2w_list: list of 4x4 numpy arrays (相机的 c2w 位姿)
#     :param scale: 视锥尺寸
#     :param color: 颜色（RGB 0~1）
#     :return: open3d.geometry.Geometry objects 可直接用 o3d.visualization.draw_geometries 显示
#     """
#     geometries = []

#     # 相机坐标系下的 frustum 顶点（小金字塔）
#     frustum = np.array([
#         [0, 0, 0],  # 相机中心
#         [-1, -1, 1],  # 左下
#         [1, -1, 1],  # 右下
#         [1, 1, 1],  # 右上
#         [-1, 1, 1]  # 左上
#     ]).T * scale  # (3, 5)

#     frustum_hom = np.vstack((frustum, np.ones((1, frustum.shape[1]))))  # (4, 5)

#     for c2w in c2w_list:
#         pts_w = (c2w @ frustum_hom)[:3, :]  # 世界坐标下的5个点
#         points = [pts_w[:, i] for i in range(5)]

#         # 创建线段连接视锥边
#         lines = [
#             [0, 1], [0, 2], [0, 3], [0, 4],  # 从中心到四角
#             [1, 2], [2, 3], [3, 4], [4, 1]  # 外框四边
#         ]
#         colors = [color for _ in lines]

#         line_set = o3d.geometry.LineSet(
#             points=o3d.utility.Vector3dVector(points),
#             lines=o3d.utility.Vector2iVector(lines)
#         )
#         line_set.colors = o3d.utility.Vector3dVector(colors)

#         geometries.append(line_set)
#     coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
#     geometries.append(coordinate_frame)
#     return geometries

# def visualize_gaussian_point_cloud(pc):
#     means3D = pc.get_xyz.detach().cpu().numpy()  # 获取高斯点的世界坐标
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(means3D)
#     return pcd