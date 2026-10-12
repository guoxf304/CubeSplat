# Local reproduction of Cube-Splat

This record separates environment validation from reproducing the paper's full
benchmark tables. The supplied SynPano `apartment` archive contains 217 frames
and is extracted into `datasets/SynPano/apartment/`. Each image is 1920 × 960.

## Environment

| Component | Tested version |
| --- | --- |
| Operating system | Ubuntu 22.04.5 LTS, x86-64 |
| Python | 3.9.25 |
| PyTorch | 2.0.1+cu118 |
| torchvision | 0.15.2+cu118 |
| CUDA toolkit / nvcc | 11.8 / V11.8.89 |
| GCC | 11.4.0 |
| NVIDIA driver | 570.211.01 |
| GPU selected for this run | RTX 3090, 24 GB |
| NumPy | 1.26.4 |
| Open3D | 0.17.0 |
| py360convert | 1.0.4 |

The environment is adapted from MonoGS. CUDA 11.8 / PyTorch 2.0.1 match the local
compiler, and the Python dependency set includes the panoramic projection and
UniFuse inference runtime. `requirements.txt` pins the main dependencies;
`requirements-lock.txt` records all installed Python package versions except the
two local extensions, which must be compiled from source.

## Installation and checks performed

Installation and validation were performed in a new environment rather than
reusing an existing MonoGS installation. The following commands recreate the
final validated dependency set and its checks:

```bash
conda create -n cubesplat python=3.9.25 pip=24.3.1 -y
conda activate cubesplat
python -m pip install -r requirements.txt

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4
bash scripts/build_extensions.sh
python -m pip check

CUDA_VISIBLE_DEVICES=0 python scripts/check_environment.py \
  --depth-image datasets/SynPano/apartment/images/0000.png
python slam.py --help
```

Both extensions were rebuilt from the bundled Cube-Splat sources after moving
aside old build directories. The final dependency check reports
`No broken requirements found.` The extension check passes all four face renders
and backpropagation into the shared pose, as well as the KNN kernel. The depth
check produces a finite, positive 512 × 1024 prediction from the first apartment
frame. The loaded model is reused on repeated calls.

The checkpoint is `UniFuse_SpatialAudioGen.pth`, with SHA-256:

```text
b36bec78f5b6860f49363b87a93e4c3a01260942bb59c5bed0e20a0a5a2d5f24
```

All 15 retained YAML configurations were loaded with their inherited base
settings. The configuration set covers SynPano, OmniBlender, and PALVIO only.
The entry point and four-face CUDA check also passed in a clean source export
containing only Git-tracked files and the new publishable files, with the ignored
training modules, weights, datasets, and old build outputs excluded.

## Dataset run

The full supplied apartment sequence is run using:

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
  python -u slam.py --config configs/ERP/SynPano/example_scene.yaml
```

The scene YAML points to `datasets/SynPano/apartment/`. Its inherited settings use
512 × 512 cubemap faces, 400 initialization iterations, up to 30 tracking
iterations, and 50 mapping iterations. GUI, W&B logging, submaps, and final
rendering evaluation are disabled. Four faces contribute to tracking and mapping.
The run uses the predicted depth prior, with no active quantized parameter list.

The process completed successfully (exit code 0) and saved the Gaussian map and
trajectory evaluation outputs. The final measurements are:

| Measurement | Result |
| --- | --- |
| Processed frames | 217 |
| Keyframes | 49 |
| Total time reported by the implementation | 380.9323 seconds |
| Overall FPS reported by the implementation | 0.5697 |
| Keyframe ATE RMSE (scale aligned) | 0.070821 m |
| All-frame ATE RMSE (scale aligned) | 0.095368 m |
| PSNR / SSIM / LPIPS | Not measured in this run |

The output directory is
`results/SynPano_apartment/2026-10-12-09-32-49/`. It contains `config.yml`,
`results.txt`, trajectory JSON/PNG files under `plot/`, and
`point_cloud/final/point_cloud.ply`. The exported PLY was read back successfully;
positions, scales, and opacity are finite. Raw local installation/build/check/run
logs and `summary.json` are kept under `results/reproduction/`, which is excluded
from Git.

The installed evo/Matplotlib combination used the existing simple-trajectory
plot fallback after a colorbar warning. The program also logged the disabled
submap merge and a CUDA IPC cleanup warning at shutdown. These messages did not
prevent the successful exit or the checked output files.

## Scope of the validation

The CUDA check establishes that the modified extension's API and forward/backward
paths execute with finite gradients. It is not a finite-difference proof of the
adjoint implementation. The apartment run validates dataset loading, depth
initialization, tracking, mapping, trajectory evaluation, and model export.
OmniBlender and PALVIO have configuration validation but have not been run locally
in this reproduction because their sequences are not present here.

The timing comes from a shared GPU machine and should not be compared directly
to the paper's hardware or benchmark timing. Reproducing PSNR/SSIM/LPIPS and the
paper tables requires the complete benchmark data and evaluation settings.
`--eval` additionally performs the implementation's 26,000 color refinement
iterations; that path is distinct from the tracking/mapping run documented here.
