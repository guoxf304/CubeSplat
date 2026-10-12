"""Check CUDA extensions, four-face rendering/backpropagation, and optional depth inference."""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from diff_gaussian_rasterization import GaussianRasterizationSettings
from simple_knn._C import distCUDA2
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from utils.camera_utils import Camera
from utils.config_utils import load_config
from utils.cubemap import ERPToCube


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-image", type=Path, help="Also run UniFuse on this ERP image")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Cube-Splat requires a CUDA-capable GPU and a CUDA PyTorch build")
    if "rel_R" not in GaussianRasterizationSettings._fields:
        raise RuntimeError("Install Cube-Splat's bundled rasterizer; the upstream API is different")
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    xyz = torch.tensor(
        [[0.15, 0.1, 2.0], [0.15, 0.1, -2.0], [2.0, 0.1, 0.15], [-2.0, 0.1, 0.15]],
        dtype=torch.float32, device="cuda",
    )
    distances = distCUDA2(xyz)
    assert torch.isfinite(distances).all() and (distances > 0).all()
    config = load_config("configs/ERP/SynPano/example_scene.yaml")
    config["Training"]["kmeans_params"]["enabled"] = False
    model = GaussianModel(0, config=config)
    model._xyz = nn.Parameter(xyz)
    model._features_dc = nn.Parameter(torch.full((4, 1, 3), 0.5, device="cuda"))
    model._features_rest = nn.Parameter(torch.empty((4, 0, 3), device="cuda"))
    model._scaling = nn.Parameter(torch.full((4, 3), -2.3, device="cuda"))
    model._rotation = nn.Parameter(torch.tensor([[1., 0., 0., 0.]] * 4, device="cuda"))
    model._opacity = nn.Parameter(torch.ones((4, 1), device="cuda"))
    size = 32
    projection = getProjectionMatrix2(
        znear=0.01, zfar=100., fx=size / 2, fy=size / 2,
        cx=size / 2, cy=size / 2, W=size, H=size,
    ).transpose(0, 1)
    cube = ERPToCube(SimpleNamespace(face_size=size))
    camera = Camera(
        0, torch.zeros((3, 64, 128), device="cuda"), None,
        torch.eye(4, device="cuda"), projection, size / 2, size / 2,
        size / 2, size / 2, torch.pi / 2, torch.pi / 2, size, size,
        cube, dataset_type="ERP",
    )
    pipeline = SimpleNamespace(**config["pipeline_params"])
    background = torch.zeros(3, device="cuda")
    for face in camera.keep_faces:
        camera.zero_grad(set_to_none=True)
        model._xyz.grad = None
        output = render(camera, model, pipeline, background, face_key=face, use_quantized=False)
        assert output["render"].shape == (3, size, size)
        assert output["visibility_filter"].any(), f"No Gaussian visible on {face}"
        output["render"].square().mean().backward()
        for label, parameter in [
            ("Gaussian positions", model._xyz),
            ("pose rotation", camera.cam_rot_delta),
            ("pose translation", camera.cam_trans_delta),
        ]:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), (face, label)
        assert camera.cam_trans_delta.grad.abs().sum() > 0, face
        print(f"PASS: {face} rendering and shared-pose gradients")
    if args.depth_image:
        import numpy as np
        from depth_em.depth_estimator import estimate_depth, load_model
        depth = estimate_depth(str(args.depth_image))
        assert depth.shape == (512, 1024) and np.isfinite(depth).all() and (depth > 0).all()
        assert load_model() is load_model(), "Depth model should be cached between keyframes"
        print(f"PASS: UniFuse inference {depth.shape}, depth range [{depth.min():.3f}, {depth.max():.3f}]")
    torch.cuda.synchronize()
    print("Environment check passed.")


if __name__ == "__main__":
    main()
