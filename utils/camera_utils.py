import torch
from torch import nn
import math

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.slam_utils import image_gradient, image_gradient_mask


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        Cube,
        device="cuda:0",
        path = None,
        dataset_type=None,
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.Cube = Cube
        self.device = device
        self.path = path
        # 数据集类型（例如 PAL 或 ERP），用于后续区分行为
        self.dataset_type = dataset_type

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.Cubemap_image = Cube.convert(self.original_image)
        self.keep_faces = ['front','left','right','back']  # 要保留的  ['front', 'left', 'back', 'right']
        self.Cubemap_image = {k: v for k, v in self.Cubemap_image.items() if k in self.keep_faces}
        # depth可以是None，也可以是字典（包含6个面的深度信息）
        self.depth = depth  # 如果是字典，格式为 {'front': tensor, 'back': tensor, ...}
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix, Cube):
        gt_color, gt_depth, gt_pose, gt_path = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx_face,
            dataset.cy_face,
            dataset.fovx,
            dataset.fovy,
            dataset.hCube,
            dataset.wCube,
            Cube,
            device=dataset.device,
            path = gt_path,
            dataset_type=getattr(dataset, "dataset_type", None),
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W, Cube):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W, Cube
        )

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def compute_grad_mask(self, config):
        '''计算图像掩码'''
        edge_threshold = config["Training"]["edge_threshold"]
        self.grad_mask = {}
        for key, img in self.Cubemap_image.items():
            gray_img = img.mean(dim=0, keepdim=True)  # 转成灰度图, torch.Size([1, H, W])
            gray_grad_v, gray_grad_h = image_gradient(gray_img)  # 计算垂直、水平梯度
            mask_v, mask_h = image_gradient_mask(gray_img)  # 计算有效区域掩码
            gray_grad_v = gray_grad_v * mask_v
            gray_grad_h = gray_grad_h * mask_h

            # 计算梯度强度（Sobel幅度）
            img_grad_intensity = torch.sqrt(gray_grad_v ** 2 + gray_grad_h ** 2)

            # 计算整张图的梯度中位数
            median_img_grad_intensity = img_grad_intensity.median()

            # 高于 (中位数 × 阈值) 的地方标记为1，否则为0
            grad_mask = img_grad_intensity > (median_img_grad_intensity * edge_threshold)

            self.grad_mask[key] = grad_mask  # 存到字典里

    def clean(self):
        self.original_image = None
        self.Cubemap_image = None
        #self.Cube = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None


    def get_cubemap_relative_rotation(self):
        return self.Cube.rotation

    def get_face_view_transform(self, face_key):
        dtype = torch.float32
        device = self.R.device

        relative_rotations = self.get_cubemap_relative_rotation()

        rel_R = relative_rotations[face_key].to(dtype=dtype, device=device)

        R = self.R.to(dtype=dtype)  # 强制转为 float32
        T = self.T.to(dtype=dtype)  # 同样也转一下

        # 构建4x4齐次变换矩阵
        W2C = getWorld2View2(rel_R @ R, rel_R @ T).transpose(0, 1)

        # 得到 world_view_transform
        return W2C

    def get_face_proj_transform(self, face_key):
        return (
            self.get_face_view_transform(face_key = face_key).unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    def get_face_proj_raw_transform(self, face_key):
        # 默认各面原始投影矩阵一样，直接返回
        return self.projection_matrix

    def get_face_camera_center(self, face_key):
        W2C = self.get_face_view_transform(face_key = face_key)
        
        # 检查矩阵是否包含 NaN 或 Inf
        if torch.isnan(W2C).any() or torch.isinf(W2C).any():
            # 如果包含 NaN 或 Inf，使用默认值
            print(f"Warning: W2C matrix contains NaN or Inf for face {face_key}, using default camera center")
            return torch.zeros(3, device=W2C.device, dtype=W2C.dtype)
        
        # 检查矩阵是否奇异（行列式接近0）
        try:
            # 对于齐次变换矩阵，我们只需要检查左上角3x3部分
            R_part = W2C[:3, :3]
            det = torch.det(R_part)
            if torch.abs(det) < 1e-6:
                # 矩阵接近奇异，使用伪逆
                print(f"Warning: W2C matrix is near-singular (det={det:.6e}) for face {face_key}, using pseudo-inverse")
                W2C_inv = torch.linalg.pinv(W2C)
            else:
                W2C_inv = W2C.inverse()
            
            camera_center = W2C_inv[3, :3]
            return camera_center
        except torch._C._LinAlgError as e:
            # 如果求逆失败，尝试使用伪逆
            print(f"Warning: Failed to invert W2C matrix for face {face_key}, using pseudo-inverse: {e}")
            try:
                W2C_inv = torch.linalg.pinv(W2C)
                camera_center = W2C_inv[3, :3]
                return camera_center
            except Exception as e2:
                # 如果伪逆也失败，返回默认值
                print(f"Error: Failed to compute pseudo-inverse for face {face_key}: {e2}, using default camera center")
                return torch.zeros(3, device=W2C.device, dtype=W2C.dtype)

    def get_face_cam_rot_delta(self, face_key):
        # 所有面共享同一套旋转增量（也可以扩展成每面独立，这里先简化）
        return self.cam_rot_delta

    def get_face_cam_trans_delta(self, face_key):
        # 所有面共享同一套平移增量
        return self.cam_trans_delta

