# Cube-Splat

**High-Fidelity 360° Gaussian Splatting SLAM via Cubemap Factorization and Adjoint-Consistent Optimization**

Xiangfei Guo, Hao Shi, Yufan Zhang, Zhonghua Yi, Yongqi Mao, Xiaoting Yin, and Kaiwei Wang

[Paper](https://arxiv.org/abs/2609.21347) · [Full text](https://arxiv.org/html/2609.21347v1)

Cube-Splat estimates camera motion and builds a dense, renderable 3D Gaussian map from panoramic image sequences. It projects each frame into four virtual pinhole views sharing one optical center. The front view defines the primary pose, and an adjoint mapping brings each view's pose gradients into that shared state. Mapping optimizes and densifies Gaussians using observations from all four views.

The implementation adapts [MonoGS / Gaussian Splatting SLAM](https://github.com/muskie82/MonoGS). The depth initialization for ERP keyframes uses the UniFuse student and checkpoint from [Depth Anywhere](https://github.com/albert100121/Depth-Anywhere).

## Supported datasets

| Dataset | Camera format | Configuration |
| --- | --- | --- |
| SynPano | Equirectangular panorama (ERP) | `configs/ERP/SynPano/example_scene.yaml` |
| OmniBlender | Equirectangular panorama (ERP) | `configs/ERP/OmniBlender/example_scene.yaml` |
| PALVIO | Panoramic annular lens (PAL) | `configs/PAL/LF-VIO/ID01.yaml` through `ID10.yaml` |

The `LF-VIO` directory contains the PALVIO camera calibration and sequence settings. Cube-Splat consumes the PAL images and pose metadata; IMU measurements are not required by this implementation. This release's configuration set is limited to these three datasets.

## Installation

### Requirements

- Linux, an NVIDIA CUDA-capable GPU, and a compatible NVIDIA driver.
- Conda, a C++ compiler, and the **CUDA 11.8 development toolkit**, including `nvcc`.
- For the tested setup: Ubuntu 22.04, GCC 11, Python 3.9, PyTorch 2.0.1 with CUDA 11.8, and an RTX 3090 (24 GB).

The environment follows MonoGS's dependency structure, with additional panoramic projection and depth inference dependencies. MonoGS's original environment uses Python 3.7 / PyTorch 1.12.1 / CUDA 11.6; this repository supplies an adapted environment for the tested CUDA 11.8 toolchain. See [MonoGS's environment](https://github.com/muskie82/MonoGS/blob/main/environment.yml) and [PyTorch's version guide](https://pytorch.org/get-started/previous-versions/) for background.

On Ubuntu, the system dependencies typically include:

```bash
sudo apt-get update
sudo apt-get install build-essential libgl1 libglib2.0-0
```

Install the CUDA toolkit using [NVIDIA's installation instructions](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-linux/index.html). The CUDA version displayed by `nvidia-smi` is the driver's supported version; check the compiler with `nvcc --version`.

### Create the Python environment

```bash
git clone https://github.com/guoxf304/CubeSplat.git
cd CubeSplat
conda env create -f environment.yml
conda activate cubesplat-repro
```

Alternatively, install the same pinned dependencies explicitly:

```bash
conda create -n cubesplat-repro python=3.9.25 pip=24.3.1 -y
conda activate cubesplat-repro
python -m pip install -r requirements.txt
```

For the complete Python package snapshot, including transitive dependencies,
install `requirements-lock.txt` instead of `requirements.txt` in that fresh
environment. The lock file targets the tested Linux / Python 3.9 / CUDA 11.8
setup; both installation paths still require building the two CUDA extensions.

### Compile Cube-Splat's CUDA extensions

**Use the sources bundled with this repository.** Cube-Splat's differentiable rasterizer has a `rel_R` setting and an adjoint mapping for per-face pose gradients. Installing the standard MonoGS or Graphdeco rasterizer can produce an incompatible API or incorrect multi-face pose gradients.

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4
bash scripts/build_extensions.sh
python -m pip check
CUDA_VISIBLE_DEVICES=0 python scripts/check_environment.py
```

Adjust `CUDA_HOME` to your toolkit installation. `build_extensions.sh` compiles both `simple-knn` and `diff-gaussian-rasterization`, with build isolation disabled so that they use the active environment's PyTorch. These directories contain bundled source, so an ordinary clone includes them.

For an RTX 3090, `export TORCH_CUDA_ARCH_LIST="8.6"` can reduce compilation time. Use the compute capability appropriate to your GPU; omitting this variable lets PyTorch choose from visible GPUs. Rebuild the extensions after changing Python, PyTorch, or CUDA. If migrating an existing checkout, remove or move aside old `submodules/*/build/` directories before rebuilding.

`check_environment.py` exercises the KNN CUDA kernel, renders each of the four cubemap faces, and backpropagates gradients into the shared camera pose. It checks execution and finite gradients; it does not establish numerical equivalence to the paper's complete optimization experiments.

## Depth checkpoint

For SynPano and OmniBlender, obtain the pretrained checkpoints from the [Depth Anywhere repository](https://github.com/albert100121/Depth-Anywhere#pretrained-weights). Place its UniFuse SpatialAudioGen checkpoint at:

```text
depth_em/checkpoints/UniFuse/UniFuse_SpatialAudioGen.pth
```

The checkpoint used in the local reproduction has SHA-256:

```text
b36bec78f5b6860f49363b87a93e4c3a01260942bb59c5bed0e20a0a5a2d5f24
```

```bash
sha256sum depth_em/checkpoints/UniFuse/UniFuse_SpatialAudioGen.pth
```

The necessary inference code is included in Git; weights and training data are excluded. The full UniFuse checkpoint contains the encoder parameters, so inference does not require a separate ImageNet encoder download. Only the UniFuse runtime is required; the other Depth Anywhere training/baseline models are not needed to run Cube-Splat.

## Prepare the data

### SynPano and OmniBlender

The current loader expects prepared images and a pose list in the following layout:

```text
datasets/
├── SynPano/
│   └── apartment/
│       ├── groundtruth.txt
│       └── images/
│           ├── 0000.png
│           └── ...
└── OmniBlender/
    └── fisher-hut/
        ├── groundtruth.txt
        └── images/
            └── ...
```

An `imgs/` directory is also accepted and takes precedence over `images/`. The ERP `groundtruth.txt` format is:

```text
# frame_id image_name x y z qx qy qz qw
0 0000.png 0.0 0.0 0.0 0.0 0.0 0.0 1.0
1 0001.png 0.1 0.0 0.0 0.0 0.0 0.0 1.0
```

The first line is a header and is skipped. Image names are relative to `images/` or `imgs/`; pose entries describe camera-to-world transforms with quaternion ordering `qx qy qz qw`. The loader converts them to world-to-camera transforms internally. Ground-truth poses are currently required by the loader and are also used for initialization and trajectory evaluation.

For a SynPano archive with `groundtruth.txt` and `images/` directly at its root:

```bash
mkdir -p datasets/SynPano/apartment
unzip datasets/apartment.zip -d datasets/SynPano/apartment
```

If the archive already contains an `apartment/` top-level folder, extract it into `datasets/SynPano/` instead. Update `Dataset.dataset_path` in the scene YAML when selecting a different scene. SynPano is introduced in the [Cube-Splat paper](https://arxiv.org/abs/2609.21347); dataset archives are distributed separately from the source tree.

OmniBlender data is available through [EgoNeRF](https://github.com/changwoonchoi/EgoNeRF#dataset). Convert its pose metadata to the format above before using this loader; the configuration expects prepared data rather than the raw EgoNeRF metadata. Set the YAML calibration width/height to the input image dimensions. `cube_face_size` controls the resolution of each virtual view (512 in the examples).

### PALVIO

Obtain PALVIO from the [LF-VIO dataset page](https://github.com/flysoaryun/LF-VIO#download-palvio-dataset). Prepare each sequence as:

```text
datasets/LF-VIO/ID02/
├── rgb.txt
├── groundtruth.txt
└── images/
    └── ...
```

`rgb.txt` contains `timestamp relative/image/path` rows. `groundtruth.txt` contains a header followed by `timestamp x y z qx qy qz qw` rows. The image and pose timestamps are associated with a maximum difference of 0.08 seconds, and the loader subsamples to at most approximately 32 Hz. Use the matching ID configuration so that the polynomial and affine PAL calibration is correct.

## Run Cube-Splat

Run commands from the repository root. `CUDA_VISIBLE_DEVICES` selects a physical GPU, which becomes logical `cuda:0` inside the program.

```bash
# SynPano apartment; GUI and rendering evaluation are disabled in the base config.
CUDA_VISIBLE_DEVICES=0 python slam.py --config configs/ERP/SynPano/example_scene.yaml

# OmniBlender: prepare fisher-hut or change the dataset path first.
CUDA_VISIBLE_DEVICES=0 python slam.py --config configs/ERP/OmniBlender/example_scene.yaml

# PALVIO: prepare ID02 first.
CUDA_VISIBLE_DEVICES=0 python slam.py --config configs/PAL/LF-VIO/ID02.yaml
```

The PAL base configuration enables the GUI. On a server, set `Results.use_gui: False` in the scene YAML. The program imports the visualization packages even in headless mode, so keep the GUI dependencies installed.

Configuration files recursively inherit their dataset's `base_config.yaml`. Scene YAMLs can override individual fields, for example:

```yaml
Results:
  use_gui: False
  use_wandb: False
  eval_rendering: False
Dataset:
  start_frame: 0
  end_frame: 30
```

Frame indices refer to the loaded sequence after timestamp association/subsampling, and `end_frame` is exclusive. Omit these bounds to process the whole loaded sequence.

### Evaluation

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
  python slam.py --config configs/ERP/SynPano/example_scene.yaml --eval
```

`--eval` enables saving, disables the GUI, enables rendering evaluation, and sets the W&B flag. `WANDB_MODE=disabled` keeps that run local without a W&B login. To use W&B logging, configure the account and omit this environment variable.

Trajectory evaluation reports ATE for keyframes and all available frames, with scale alignment for monocular runs. Rendering evaluation reports PSNR, SSIM, and LPIPS per face and overall. The evaluation path also performs **26,000 color refinement iterations**, so it takes substantially longer than a tracking/mapping-only run. LPIPS can download pretrained perceptual-network weights on its first use.

Results are stored beneath `results/<dataset>_<scene>/<timestamp>/`, including the resolved configuration, `results.txt`, trajectory evaluation files, and `point_cloud/final/point_cloud.ply`. Rendering evaluation adds before/after refinement metrics and a refined model under `point_cloud/final_after_opt/`. Quantization and submap output are controlled separately by the configuration; the default ERP quantization parameter list is empty.

## Local reproduction

The local verification uses the environment above and the supplied SynPano apartment sequence (217 images at 1920 × 960). Both CUDA extensions, all four face render/backward checks, and UniFuse inference passed. The full 217-frame tracking/mapping run completed and exported the Gaussian PLY model, with 49 keyframes, keyframe ATE of **0.070821 m**, and all-frame ATE of **0.095368 m** (monocular scale alignment). It reported **380.93 s / 0.5697 FPS** on a shared RTX 3090 machine. PSNR/SSIM/LPIPS and final color refinement were not run in this validation. Detailed results and the exact validation commands are recorded in [docs/reproduction.md](docs/reproduction.md).

To validate the depth path after preparing the data and checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/check_environment.py \
  --depth-image datasets/SynPano/apartment/images/0000.png
```

Reproducing the paper's benchmark tables additionally requires the full benchmark sequences, the corresponding experiment settings, and rendering evaluation. GPU sharing and asynchronous frontend/backend scheduling can affect timing and the numerical trajectory. A short installation check should not be compared to the paper's full-sequence results.

## Code structure

| Path | Purpose |
| --- | --- |
| `slam.py` | Entry point, process orchestration, and final evaluation |
| `utils/slam_frontend.py` | Photometric tracking and keyframe selection |
| `utils/slam_backend.py` | Gaussian insertion, map optimization, pruning, and refinement |
| `utils/cubemap.py` | ERP/PAL projection and fixed relative face rotations |
| `gaussian_splatting/` | Gaussian representation and Python rendering interface |
| `submodules/` | Bundled KNN and modified differentiable CUDA rasterizer |
| `depth_em/` | Depth Anywhere / UniFuse inference runtime |
| `configs/` | SynPano, OmniBlender, and PALVIO experiment settings |

## Acknowledgements and licenses

We build on [MonoGS](https://github.com/muskie82/MonoGS), [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), the [pose-aware rasterizer](https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose), [Depth Anywhere](https://github.com/albert100121/Depth-Anywhere), [UniFuse](https://github.com/alibaba/UniFuse-Unidirectional-Fusion), [py360convert](https://github.com/sunset1995/py360convert), and the [Gaussian Splatting Viewer](https://github.com/limacv/GaussianSplattingViewer).

Third-party components retain their original terms. See `gaussian_splatting/LICENSE.md`, `submodules/diff-gaussian-rasterization/LICENSE.md`, `gui/gl_render/LICENSE`, and `depth_em/licenses/`. The inherited Gaussian Splatting and MonoGS terms include research/non-commercial restrictions; this README does not grant a blanket permissive or commercial license to the combined code. Check upstream terms for the versions used before redistributing or using their components commercially.

## Citation

```bibtex
@article{guo2026cubesplat,
  title={Cube-Splat: High-Fidelity 360{\textdegree} Gaussian Splatting SLAM via Cubemap Factorization and Adjoint-Consistent Optimization},
  author={Guo, Xiangfei and Shi, Hao and Zhang, Yufan and Yi, Zhonghua and Mao, Yongqi and Yin, Xiaoting and Wang, Kaiwei},
  journal={arXiv preprint arXiv:2609.21347},
  year={2026},
  url={https://arxiv.org/abs/2609.21347}
}
```
