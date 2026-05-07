# 在导入PIL之前设置环境变量，避免Tkinter问题
import os
os.environ['PIL_USE_TKINTER'] = '0'  # 禁用PIL的Tkinter支持

import csv
import glob
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

from gaussian_splatting.utils.graphics_utils import focal2fov

try:
    import pyrealsense2 as rs
except Exception:
    pass


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass


class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        self.calibration = config["Dataset"]["Calibration"]


    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        # 显式关闭图像文件以避免 tkinter 相关警告
        with Image.open(color_path) as img:
            image = np.array(img.convert("RGB"))
        
        image = ( # 转化为 cuda
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        # ERP图像在经纬度投影下左右边界周期连续，按角度转换为像素后做水平循环平移
        if getattr(self, "dataset_type", None) == "ERP" and getattr(self, "erp_shift_enabled", True):
            shift_deg = float(getattr(self, "erp_shift_deg", 45.0))
            width = image.shape[-1]
            shift_px = int(round((shift_deg / 360.0) * width))
            if shift_px != 0:
                image = torch.roll(image, shifts=shift_px, dims=-1)

        pose = torch.from_numpy(pose).to(device=self.device) # cuda
        return image, None, pose, color_path

class PALParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            k = np.argmin(np.abs(tstamp_pose - t))

            if  np.abs(tstamp_pose[k] - t) < max_dt:
                associations.append((i, k))
        return associations

    def load_poses(self, datapath, frame_rate=-1):

        pose_list = os.path.join(datapath, "groundtruth.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        # depth_list = os.path.join(datapath, "depth.txt")

        image_data = self.parse_list(image_list)
        # depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        # tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        tstamp_depth = None
        associations = self.associate_frames(tstamp_image, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.frames = [], [], []

        for ix in indicies:
            (i, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            # self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]

            quat = pose_vecs[k][4:]
            trans = pose_vecs[k][1:4]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            self.poses += [np.linalg.inv(T)]

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }

            self.frames.append(frame)


class OmniPhotosParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def load_poses(self, datapath, frame_rate=-1):
        """加载OmniPhotos数据集的位姿信息（JSON格式）
        
        数据结构：
        - data_extrinsics.json: {"extrinsics": [{"key": pose_id, "value": {"rotation": [...], "center": [...]}}]}
        - data_views.json: {"views": [{"key": view_id, "value": {"ptr_wrapper": {"data": {...}}}}]}
        """
        extrinsics_file = os.path.join(datapath, "data_extrinsics.json")
        views_file = os.path.join(datapath, "data_views.json")
        
        # 加载JSON文件
        with open(extrinsics_file, 'r') as f:
            extrinsics_data = json.load(f)
        with open(views_file, 'r') as f:
            views_data = json.load(f)
        
        # 解析位姿数据：extrinsics是一个列表，每个元素是 {"key": pose_id, "value": {"rotation": [...], "center": [...]}}
        poses_dict = {}
        for item in extrinsics_data["extrinsics"]:
            pose_id = item["key"]
            pose_value = item["value"]
            rotation = np.array(pose_value["rotation"])
            center = np.array(pose_value["center"])
            
            # 构建4x4变换矩阵（world to camera），然后转换为camera to world
            T = np.eye(4)
            T[:3, :3] = rotation
            T[:3, 3] = center
            T_c2w = np.linalg.inv(T)
            poses_dict[pose_id] = T_c2w
        
        # 解析视图数据：views是一个列表，每个元素是 {"key": view_id, "value": {"ptr_wrapper": {"data": {...}}}}
        views_list = []
        for item in views_data["views"]:
            view_value = item["value"]
            # 实际数据在 ptr_wrapper.data 中
            view_data = view_value["ptr_wrapper"]["data"]
            views_list.append(view_data)
        
        # 按id_view排序
        views_list = sorted(views_list, key=lambda v: v["id_view"])
        
        # 按帧率采样
        indicies = [0]
        for i in range(1, len(views_list)):
            if frame_rate > 0 and (i - indicies[-1]) >= 1.0 / frame_rate:
                indicies.append(i)
            elif frame_rate <= 0:
                indicies.append(i)
        
        # 构建路径和位姿列表
        self.color_paths, self.poses, self.frames = [], [], []
        self.depth_paths = []
        
        # 获取root_path（如果存在）
        root_path = views_data.get("root_path", "")
        if root_path and os.path.exists(root_path):
            image_base_dir = root_path
        else:
            image_base_dir = datapath
        
        for idx in indicies:
            view_data = views_list[idx]
            filename = view_data["filename"]
            id_pose = view_data["id_pose"]
            
            # 构建图像路径
            color_path = os.path.join(image_base_dir, filename)
            if not os.path.exists(color_path):
                color_path = os.path.join(datapath, filename)
            if not os.path.exists(color_path):
                color_path = os.path.join(datapath, "images", filename)
            
            if not os.path.exists(color_path):
                print(f"Warning: Image not found: {filename}")
                continue
            
            self.color_paths.append(color_path)
            
            # 获取对应的位姿
            if id_pose in poses_dict:
                pose = poses_dict[id_pose]
            else:
                print(f"Warning: Pose ID {id_pose} not found, using identity matrix")
                pose = np.eye(4)
            
            self.poses.append(pose)
            self.frames.append({
                "file_path": str(color_path),
                "transform_matrix": pose.tolist(),
            })
            
            # OmniPhotos没有深度信息
            depth_dict = {face: None for face in ['front', 'back', 'left', 'right', 'top', 'bottom']}
            self.depth_paths.append(depth_dict)


class ERPParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            k = np.argmin(np.abs(tstamp_pose - t))

            if  np.abs(tstamp_pose[k] - t) < max_dt:
                associations.append((i, k))
        return associations

    def load_poses(self, datapath, frame_rate=-1):

        pose_list = os.path.join(datapath, "groundtruth.txt")

        # image_list = os.path.join(datapath, "rgb.txt")




        pose_data = self.parse_list(pose_list, skiprows=1)

        image_data = pose_data[:, :2]

        pose_vecs = pose_data[:, 0:]

        tstamp_image = image_data[:, 0].astype(np.float64)

        tstamp_pose = pose_data[:, 0].astype(np.float64)
        tstamp_depth = None
        associations = self.associate_frames(tstamp_image, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.frames = [], [], []
        self.depth_paths = []  # 存储深度路径（字典列表，每个字典包含6个面的路径）

        for ix in indicies:
            (i, k) = associations[ix]
            # 支持 imgs 和 images 目录（优先使用 imgs，如 Ricoh360 数据集）
            img_dir = "imgs" if os.path.exists(os.path.join(datapath, "imgs")) else "images"
            color_path = os.path.join(datapath, img_dir, image_data[i, 1])
            self.color_paths += [color_path]

            quat = pose_vecs[k][5:]
            trans = pose_vecs[k][2:5]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            self.poses += [np.linalg.inv(T)]

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }

            self.frames.append(frame)
            
            # 360VO数据集没有深度信息
            depth_dict = {face: None for face in ['front', 'back', 'left', 'right', 'top', 'bottom']}
            self.depth_paths.append(depth_dict)




class PALDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)

        # 数据集类型标记
        self.dataset_type = "PAL"


        # Camera prameters
        self.face_size = self.calibration["cube_face_size"]
        # self.fx = calibration["fx"]
        self.fx = self.face_size / 2
        # self.fy = calibration["fy"]
        self.fy = self.face_size / 2

        # self.wCube = calibration["widthCube"]
        self.wCube = self.face_size
        # self.hCube = calibration["heightCube"]
        self.hCube = int(175 * self.face_size / 256)
        # self.cx_face = calibration["cx_face"]
        self.cx_face = self.face_size / 2
        # self.cy_face = calibration["cy_face"]
        self.cy_face = self.face_size / 2
        self.width = self.calibration["width"]
        self.height = self.calibration["height"]
        self.cx = self.calibration["cx"]
        self.cy = self.calibration["cy"]
        self.fovx = focal2fov(self.fx, self.wCube)
        self.fovy = focal2fov(self.fy, self.wCube)

        # depth parameters
        self.has_depth = True if "depth_scale" in self.calibration.keys() else False
        self.depth_scale = self.calibration["depth_scale"] if self.has_depth else None

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

        dataset_path = config["Dataset"]["dataset_path"]
        parser = PALParser(dataset_path) # 待更改
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        # self.depth_paths = parser.depth_paths
        self.poses = parser.poses

        self.poly_parameters = config["Dataset"]["poly_parameters"]
        self.inv_poly_parameters = config["Dataset"]["inv_poly_parameters"]
        self.affine_parameters = config["Dataset"]["affine_parameters"]

class ERPDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]

        # 数据集类型标记
        self.dataset_type = "ERP"

        self.cube_face_size_base = int(self.calibration["cube_face_size"])
        self.erp_face_fov_deg_train = float(config["Dataset"].get("erp_face_fov_deg_train", 95.0))
        self.erp_face_fov_deg_eval_metric = float(config["Dataset"].get("erp_face_fov_deg_eval_metric", 90.0))
        self.f_base = self.cube_face_size_base / 2.0
        train_fov_rad = np.deg2rad(self.erp_face_fov_deg_train)
        self.train_face_size = int(round(2.0 * self.f_base * np.tan(train_fov_rad / 2.0)))
        self.face_size = self.train_face_size
        self.fx = self.f_base
        self.fy = self.f_base
        self.wCube = self.train_face_size
        self.hCube = self.train_face_size
        self.cx_face = self.wCube / 2.0
        self.cy_face = self.hCube / 2.0
        self.width = self.calibration["width"]
        self.height = self.calibration["height"]
        self.fovx = focal2fov(self.fx, self.wCube)
        self.fovy = focal2fov(self.fy, self.wCube)
        self.has_depth = True if "depth_scale" in self.calibration.keys() else False
        self.erp_shift_enabled = config["Dataset"].get("erp_shift_enabled", True)
        self.erp_shift_deg = float(config["Dataset"].get("erp_shift_deg", 45.0))

        # 根据数据集类型选择不同的parser
        dataset_type = config["Dataset"].get("parser_type", "ERP")  # 默认使用ERPParser
        if dataset_type == "OmniPhotos":
            parser = OmniPhotosParser(dataset_path)
        else:
            parser = ERPParser(dataset_path)
        
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths  # 深度路径（字典列表）
        self.poses = parser.poses




def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "LF-VIO":
        return PALDataset(args, path, config)
    elif config["Dataset"]["type"] == "LF-VISLAM":
        return PALDataset(args, path, config)
    elif config["Dataset"]["type"] == "ERP":
        return ERPDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")
