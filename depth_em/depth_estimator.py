"""
深度估计函数模块
输入全景图片，返回对应的深度全景图片
"""

import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import Union

# 路径设置：确保可从项目根目录调用
# 获取 depth_em 目录的绝对路径
DEPTH_EM_DIR = os.path.dirname(os.path.abspath(__file__))
from depth_em.baseline_models.UniFuse.networks import UniFuse
from depth_em.projection import Equirec2Cube as py360_E2C

# 图像归一化参数
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 默认模型配置
DEFAULT_MODEL_CONFIG = {
    'num_layers': 18,
    'equi_h': 512,
    'equi_w': 1024,
    # The complete UniFuse checkpoint already includes the encoder weights.
    'pretrained': False,
    'max_depth': 10.0,
    'fusion_type': 'cee',
    'se_in_fusion': True
}

# 全局变量，用于缓存模型（避免重复加载）
_model = None
_device = None


def load_model(ckpt_path: str = 'checkpoints/UniFuse/UniFuse_SpatialAudioGen.pth',
               device: str = None) -> torch.nn.Module:
    """
    加载UniFuse深度估计模型

    Args:
        ckpt_path: 模型权重路径，默认为 'checkpoints/UniFuse/UniFuse_SpatialAudioGen.pth'
                  如果是相对路径，将相对于 depth_em 目录解析
        device: 设备（'cuda' 或 'cpu'），如果为None则自动选择

    Returns:
        加载好的模型
    """
    global _model, _device

    # 确定设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Resolve the default device before comparing it with the cached device.
    if _model is not None and _device == device:
        return _model
    _device = device

    # 处理路径：如果是相对路径，相对于 depth_em 目录
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(DEPTH_EM_DIR, ckpt_path)

    # 检查权重文件是否存在
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"模型权重文件不存在: {ckpt_path}")

    print(f"正在加载深度估计模型: {ckpt_path}")

    # 创建模型
    model = UniFuse(**DEFAULT_MODEL_CONFIG)

    # 加载权重
    ckpt = torch.load(ckpt_path, map_location=device)

    # 处理不同的权重文件格式
    if isinstance(ckpt, dict):
        # 如果权重文件是字典，尝试不同的键
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # 加载权重（允许部分匹配，因为某些层可能不匹配）
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # 如果严格加载失败，尝试非严格加载
        print("警告: 严格加载权重失败，尝试非严格加载...")
        model.load_state_dict(state_dict, strict=False)

    # 移动到设备并设置为评估模式
    model.to(device)
    model.eval()

    _model = model
    print("模型加载完成")

    return model


def preprocess_image(image: Union[str, np.ndarray],
                     height: int = 512,
                     width: int = 1024,
                     device: str = None) -> tuple:
    """
    预处理输入的全景图片

    Args:
        image: 输入图片，可以是文件路径（str）或numpy数组（H, W, 3）
        height: 目标高度，默认512
        width: 目标宽度，默认1024
        device: 设备（'cuda' 或 'cpu'），如果为None则自动选择

    Returns:
        (normalize_rgb, normalize_cube_rgb): 处理后的ERP和cubemap tensor
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载图片
    if isinstance(image, str):
        rgb = cv2.imread(image)
        if rgb is None:
            raise ValueError(f"无法读取图片: {image}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        # 假设输入是RGB格式的numpy数组
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = image
        else:
            raise ValueError(f"输入的numpy数组格式不正确，期望 (H, W, 3)，得到 {image.shape}")
    else:
        raise TypeError(f"不支持的输入类型: {type(image)}，期望 str 或 np.ndarray")

    # 调整大小
    rgb = cv2.resize(rgb, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

    # 转换为cubemap
    E2C = py360_E2C(equ_h=height, equ_w=width, face_w=(height//2))
    cube_rgb = E2C.run(rgb)

    # 转换为tensor并归一化
    totensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=MEAN, std=STD)

    tensor_rgb = totensor(rgb)
    normalize_rgb = normalize(tensor_rgb)

    tensor_cube_rgb = totensor(cube_rgb)
    normalize_cube_rgb = normalize(tensor_cube_rgb)

    # 添加batch维度 (CHW -> BCHW)
    normalize_rgb = normalize_rgb.unsqueeze(0).to(device)
    normalize_cube_rgb = normalize_cube_rgb.unsqueeze(0).to(device)

    return normalize_rgb, normalize_cube_rgb


def estimate_depth(image: Union[str, np.ndarray],
                   ckpt_path: str = 'checkpoints/UniFuse/UniFuse_SpatialAudioGen.pth',
                   device: str = None,
                   return_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    估计全景图片的深度

    Args:
        image: 输入图片，可以是文件路径（str）或numpy数组（H, W, 3）
        ckpt_path: 模型权重路径，默认为 'checkpoints/UniFuse/UniFuse_SpatialAudioGen.pth'
        device: 设备（'cuda' 或 'cpu'），如果为None则自动选择
        return_tensor: 如果为True，返回torch.Tensor；否则返回numpy数组

    Returns:
        深度图，形状为 (H, W)，默认返回numpy数组
    """
    # 加载模型
    model = load_model(ckpt_path, device)

    # 预处理图片
    rgb, cube = preprocess_image(image, device=device)

    # 推理
    with torch.no_grad():
        outputs = model(rgb, cube)
        inv_depth = outputs['pred_depth'].squeeze()
        # 将逆深度转换为深度，并截断最大深度20
        depth = 1.0 / torch.clamp(inv_depth, min=1e-6)
        depth = torch.clamp(depth, max=21)

    # 返回结果
    if return_tensor:
        return depth
    else:
        return depth.cpu().numpy()


# 便捷函数别名
def predict_depth(image: Union[str, np.ndarray],
                  ckpt_path: str = 'checkpoints/UniFuse/UniFuse_SpatialAudioGen.pth',
                  device: str = None) -> np.ndarray:
    """
    预测深度（estimate_depth的别名）

    Args:
        image: 输入图片，可以是文件路径（str）或numpy数组（H, W, 3）
        ckpt_path: 模型权重路径
        device: 设备（'cuda' 或 'cpu'），如果为None则自动选择

    Returns:
        深度图，形状为 (H, W)，numpy数组
    """
    return estimate_depth(image, ckpt_path, device, return_tensor=False)


if __name__ == '__main__':
    # 示例用法
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"处理图片: {image_path}")

        # 估计深度
        depth = estimate_depth(image_path)

        print(f"深度图形状: {depth.shape}")
        print(f"深度值范围: [{depth.min():.4f}, {depth.max():.4f}]")

        # 保存深度图（可选）
        if len(sys.argv) > 2:
            output_path = sys.argv[2]
            np.save(output_path, depth)
            print(f"深度图已保存到: {output_path}")
    else:
        print("用法: python depth_estimator.py <输入图片路径> [输出npy路径]")
