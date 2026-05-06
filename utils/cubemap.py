
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import py360convert
import math


class PALToCubemapConverter:
    def __init__(self, dataset):
        self.dataset = dataset
        self.face_size = dataset.face_size
        self.wPAL = dataset.width  # width = 1280
        self.hPAL = dataset.height  # height = 960

        self.cx = dataset.cx
        self.cy = dataset.cy
        self.center = (self.cx, self.cy)

        PAL_shape = [self.wPAL, self.hPAL]
        self.channels = 1 if len(PAL_shape) == 2 else PAL_shape[2]

        poly_parameters = dataset.poly_parameters
        self.poly_coeff = [
            poly_parameters["p0"],
            poly_parameters["p1"],
            poly_parameters["p2"],
            poly_parameters["p3"],
            poly_parameters["p4"]
        ]

        inv_poly_parameters = dataset.inv_poly_parameters
        self.inv_poly_coeff = [
            inv_poly_parameters["p0"],
            inv_poly_parameters["p1"],
            inv_poly_parameters["p2"],
            inv_poly_parameters["p3"],
            inv_poly_parameters["p4"],
            inv_poly_parameters["p5"],
            inv_poly_parameters["p6"],
            inv_poly_parameters["p7"],
            inv_poly_parameters["p8"],
            inv_poly_parameters["p9"],
            inv_poly_parameters["p10"],
            inv_poly_parameters["p11"],
            inv_poly_parameters["p12"],
            inv_poly_parameters["p13"],
            inv_poly_parameters["p14"],
            inv_poly_parameters["p15"],
            inv_poly_parameters["p16"],
            inv_poly_parameters["p17"],
            inv_poly_parameters["p18"],
            inv_poly_parameters["p19"]
        ]
        # 初始化映射表
        self.mMap1 = {}
        self.mMap2 = {}
        self._precompute_grids()
        self.rotation = self.get_all_face_rotation()


    def _precompute_grids(self):
        face_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
        self.mGrid = {}
        for face in face_names:
            map1 = np.full((self.face_size, self.face_size), -1, dtype=np.float32)
            map2 = np.full((self.face_size, self.face_size), -1, dtype=np.float32)
            for y in range(self.face_size):
                for x in range(self.face_size):
                    nx = (x / (self.face_size - 1)) * 2 - 1
                    ny = (y / (self.face_size - 1)) * 2 - 1
                    if face == 'bottom':
                        vec = np.array([nx, ny, 1])
                    elif face == 'top':
                        vec = np.array([-nx, ny, -1])
                    elif face == 'left':
                        vec = np.array([-1, nx, ny])
                    elif face == 'right':
                        vec = np.array([1, -nx, ny])
                    elif face == 'front':
                        vec = np.array([nx, 1, ny])
                    elif face == 'back':
                        vec = np.array([-nx, -1, ny])
                    else:
                        continue
                    vec = vec / np.linalg.norm(vec)
                    fx, fy = self._vector_to_PAL_pixel(vec)
                    map1[y, x] = fx
                    map2[y, x] = fy
            # 归一化为 [-1, 1] 区间，适用于 grid_sample
            grid_x = 2.0 * map1 / (self.wPAL - 1) - 1.0
            grid_y = 2.0 * map2 / (self.hPAL - 1) - 1.0
            grid = np.stack([grid_x, grid_y], axis=-1)  # (H, W, 2)
            self.mGrid[face] = torch.from_numpy(grid).unsqueeze(0).float()  # (1, H, W, 2)

    def _vector_to_PAL_pixel(self, vec):
        x, y, z = vec
        rho = np.sqrt(x ** 2 + y ** 2)
        if rho == 0:
            u = self.center[0]
            v = self.center[1]
        else:
            theta = np.arctan2(z, rho)
            r = 0
            for i, c in enumerate(self.inv_poly_coeff):
                r += c * (theta ** i)
            u = self.center[0] + r * x / rho
            v = self.center[1] + r * y / rho
        return u, v

    def get_all_face_rotation(self):
        dtype = torch.float32  # ★★ 强制使用 float32，兼容 rasterizer
        device = 'cuda:0'
        rotations = {
            'front': torch.eye(3, dtype=dtype, device=device),
            'back': self.get_rotation_matrix('y', 180, dtype, device),
            'left': self.get_rotation_matrix('y', 90, dtype, device),
            'right': self.get_rotation_matrix('y', 270, dtype, device),
            'top': self.get_rotation_matrix('x', -90, dtype, device),
            'bottom': self.get_rotation_matrix('x', 90, dtype, device),
            '30': self.get_rotation_matrix('x', 10, dtype, device),
            '60': self.get_rotation_matrix('x', 20, dtype, device),
            '120': self.get_rotation_matrix('x', 30, dtype, device),
            '150': self.get_rotation_matrix('x', 40, dtype, device),
            '210': self.get_rotation_matrix('x', 50, dtype, device),
            '240': self.get_rotation_matrix('x', 60, dtype, device),
            '300': self.get_rotation_matrix('x', 70, dtype, device),
            '330': self.get_rotation_matrix('x', 80, dtype, device),
        }
        return rotations


    def get_rotation_matrix(self, axis, angle_degrees, dtype=torch.float32, device='cuda:0'):
        angle = math.radians(angle_degrees)
        c = math.cos(angle)
        s = math.sin(angle)

        if axis == 'x':
            R = torch.tensor([[1, 0, 0],
                              [0, c, -s],
                              [0, s, c]], dtype=dtype, device=device)
        elif axis == 'y':
            R = torch.tensor([[c, 0, s],
                              [0, 1, 0],
                              [-s, 0, c]], dtype=dtype, device=device)
        elif axis == 'z':
            R = torch.tensor([[c, -s, 0],
                              [s, c, 0],
                              [0, 0, 1]], dtype=dtype, device=device)
        else:
            raise ValueError(f"Invalid axis '{axis}', must be 'x', 'y', or 'z'.")
        return R


    def convert(self, PAL_img):
        if PAL_img is None:
            cube_faces = {}
            for face in self.mGrid:
                cube_faces[face] = torch.zeros(3, 512, 512)
            return cube_faces

        device = PAL_img.device
        C, H, W = PAL_img.shape
        # PAL_img = torch.flip(PAL_img, dims=[1, 2]) # 翻转180度
        PAL_img = PAL_img.unsqueeze(0)  # (1, C, H, W)


        cube_faces = {}
        for face in self.mGrid:
            grid = self.mGrid[face].to(device)  # (1, Hf, Wf, 2)
            out = F.grid_sample(PAL_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            cube_faces[face] = out.squeeze(0)  # (C, Hf, Wf)
            cube_faces[face] = cube_faces[face][:, :self.dataset.hCube, :]
        return cube_faces



class EquirectangularToCubemapConverter:
    def __init__(self, dataset):
        self.dataset = dataset
        self.face_size = dataset.face_size
        self.W = dataset.width
        self.H = dataset.height


        ERP_shape = [self.W, self.H]
        self.channels = 1 if len(ERP_shape) == 2 else ERP_shape[2]

        # 初始化映射表
        self.mMap1 = {}
        self.mMap2 = {}
        self._precompute_grids()

    def _precompute_grids(self):
        face_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
        self.mGrid = {}
        for face in face_names:
            map1 = np.full((self.face_size, self.face_size), -1, dtype=np.float32)
            map2 = np.full((self.face_size, self.face_size), -1, dtype=np.float32)
            for y in range(self.face_size):
                for x in range(self.face_size):
                    nx = (x / (self.face_size - 1)) * 2 - 1
                    ny = (y / (self.face_size - 1)) * 2 - 1
                    if face == 'bottom':
                        vec = np.array([nx, ny, 1])
                    elif face == 'top':
                        vec = np.array([-nx, ny, -1])
                    elif face == 'left':
                        vec = np.array([-1, nx, ny])
                    elif face == 'right':
                        vec = np.array([1, -nx, ny])
                    elif face == 'front':
                        vec = np.array([nx, 1, ny])
                    elif face == 'back':
                        vec = np.array([-nx, -1, ny])
                    else:
                        continue
                    vec = vec / np.linalg.norm(vec)
                    fx, fy = self._vector_to_ERP_pixel(vec)
                    map1[y, x] = fx
                    map2[y, x] = fy
            # 归一化为 [-1, 1] 区间，适用于 grid_sample
            grid_x = 2.0 * map1 / (self.W - 1) - 1.0
            grid_y = 2.0 * map2 / (self.H - 1) - 1.0
            grid = np.stack([grid_x, grid_y], axis=-1)  # (H, W, 2)
            self.mGrid[face] = torch.from_numpy(grid).unsqueeze(0).float()  # (1, H, W, 2)

    def _vector_to_ERP_pixel(self, vec):
        x, y, z = vec
        theta = np.arctan2(y, x)
        phi = np.arcsin(z)

        u = (theta / (np.pi * 2.0) + 0.5) * self.W
        v = (phi / np.pi + 0.5) * self.H
        return u, v

    def convert(self, ERP_img):
        if ERP_img is None:
            cube_faces = {}
            for face in self.mGrid:
                cube_faces[face] = torch.zeros(3, 512, 512)
            return cube_faces

        device = ERP_img.device

        ERP_img = ERP_img.unsqueeze(0)  # (1, C, H, W)


        cube_faces = {}
        for face in self.mGrid:
            grid = self.mGrid[face].to(device)  # (1, Hf, Wf, 2)
            out = F.grid_sample(ERP_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            cube_faces[face] = out.squeeze(0)  # (C, Hf, Wf)
            cube_faces[face] = cube_faces[face][:, :, :]
        return cube_faces




class ERPToCube:
    def __init__(self, dataset):
        self.dataset = dataset
        self.face_size = dataset.face_size
        self.rotation = self.get_all_face_rotation()

    def get_all_face_rotation(self):
        dtype = torch.float32  # ★★ 强制使用 float32，兼容 rasterizer
        device = 'cuda:0'
        rotations = {
            'front': torch.eye(3, dtype=dtype, device=device),
            'back': self.get_rotation_matrix('y', 180, dtype, device),
            'left': self.get_rotation_matrix('y', 90, dtype, device),
            'right': self.get_rotation_matrix('y', 270, dtype, device),
            'top': self.get_rotation_matrix('x', -90, dtype, device),
            'bottom': self.get_rotation_matrix('x', 90, dtype, device),
            '30': self.get_rotation_matrix('x', 10, dtype, device),
            '60': self.get_rotation_matrix('x', 20, dtype, device),
            '120': self.get_rotation_matrix('x', 30, dtype, device),
            '150': self.get_rotation_matrix('x', 40, dtype, device),
            '210': self.get_rotation_matrix('x', 50, dtype, device),
            '240': self.get_rotation_matrix('x', 60, dtype, device),
            '300': self.get_rotation_matrix('x', 70, dtype, device),
            '330': self.get_rotation_matrix('x', 80, dtype, device),
        }
        return rotations


    def get_rotation_matrix(self, axis, angle_degrees, dtype=torch.float32, device='cuda:0'):
        angle = math.radians(angle_degrees)
        c = math.cos(angle)
        s = math.sin(angle)

        if axis == 'x':
            R = torch.tensor([[1, 0, 0],
                              [0, c, -s],
                              [0, s, c]], dtype=dtype, device=device)
        elif axis == 'y':
            R = torch.tensor([[c, 0, s],
                              [0, 1, 0],
                              [-s, 0, c]], dtype=dtype, device=device)
        elif axis == 'z':
            R = torch.tensor([[c, -s, 0],
                              [s, c, 0],
                              [0, 0, 1]], dtype=dtype, device=device)
        else:
            raise ValueError(f"Invalid axis '{axis}', must be 'x', 'y', or 'z'.")
        return R
    def convert(self, ERP_tensor):
        if ERP_tensor is None:
            cube_faces = {}
            face_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
            for face in face_names:
                cube_faces[face] = torch.zeros(3, self.face_size, self.face_size)
            return cube_faces

        device = ERP_tensor.device
        ERP_img = ERP_tensor.permute(1, 2, 0).cpu().numpy()
        c_img = py360convert.e2c(ERP_img, face_w=self.face_size, mode='bicubic', cube_format='dict')

        cube_faces = {
            'front': torch.from_numpy(c_img['F']).permute(2, 0, 1).float().to(device),
            'back': torch.from_numpy(c_img['B']).permute(2, 0, 1).float().to(device),
            'left': torch.from_numpy(c_img['L']).permute(2, 0, 1).float().to(device),
            'right': torch.from_numpy(c_img['R']).permute(2, 0, 1).float().to(device),
            'top': torch.from_numpy(c_img['U']).permute(2, 0, 1).float().to(device),
            'bottom': torch.from_numpy(c_img['D']).permute(2, 0, 1).float().to(device),
        }
        return cube_faces














def plot_cubemap(PAL_img, cubemap):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 4, 1)
    if len(PAL_img.shape) == 2 or PAL_img.shape[2] == 1:
        plt.imshow(PAL_img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(PAL_img, cv2.COLOR_BGR2RGB))
    plt.title('PAL Image')

    faces = ['front', 'right', 'back', 'left', 'top', 'bottom']
    for i, face in enumerate(faces):
        plt.subplot(2, 4, i + 2)
        img = cubemap[face]
        if img.shape[-1] == 1:
            img = img.squeeze()
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(face.capitalize())
    plt.tight_layout()
    plt.show()

def tensor_plot(tensor1, tensor2, frame_id=0, save_dir='output'):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

    # 第一张图（render）
    axes[0].imshow(tensor1.permute(1, 2, 0).cpu().detach().numpy())
    axes[0].set_title(f'Render View - Frame {frame_id}')
    axes[0].axis('off')

    # 第二张图（ground truth）
    axes[1].imshow(tensor2.permute(1, 2, 0).cpu().detach().numpy())
    axes[1].set_title(f'Ground Truth View - Frame {frame_id}')
    axes[1].axis('off')

    # 布局优化
    plt.tight_layout()
    plt.show()

    # 保存图像
    filename = f'{save_dir}/frame_{frame_id:04d}.png'
    #plt.savefig(filename)
    #plt.close(fig)  # 释放内存


