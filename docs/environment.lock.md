# Environment Lock

## Validation Date

2026-08-30 (Asia/Shanghai)

This file freezes the WSL machine and environment that were actually
validated for Phase 0.

## Host / WSL

| Component | Value |
|---|---|
| Host OS | Windows 10 build 10.0.19045.6466 |
| WSL | 2.7.12.0 |
| WSL kernel | 6.18.33.2-microsoft-standard-WSL2 |
| Ubuntu | 24.04.4 LTS (Noble) |
| Architecture | x86_64 |

## Hardware / NVIDIA

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 |
| VRAM | 6144 MiB |
| NVIDIA-SMI | 590.74 |
| NVIDIA driver | 592.82 |
| Driver-reported CUDA capability | 13.1 |
| Compute capability | 7.5 (sm_75) |

The 13.1 value is the CUDA capability reported by the Windows driver through
WSL. It is not the CUDA Toolkit version and is not torch.version.cuda. No Linux
NVIDIA display driver was installed in WSL.

## Python

| Component | Value |
|---|---|
| Environment manager | venv |
| Environment name/path | .venv at the repository root |
| Python | 3.12.3 |
| pip | 26.2.1 |
| Conda / Mamba | not installed |

## Build Tools

| Component | Value | Status |
|---|---|---|
| System GCC / G++ | 13.3.0 | Present |
| Build GCC / G++ | 12.4.0 | Used for OpenPCDet |
| CMake | 3.28.3 | Present |
| Ninja | 1.13.0 | Present in .venv |
| Make | 4.3 | Present |
| nvcc | 12.0.140 | User-space Toolkit extraction |

Ubuntu 24.04 provides GCC 13.3, which the extracted CUDA 12.0 compiler rejects
by default. GCC 12.4 was unpacked under the ignored .gcc-12 directory. CUDA
12.0 was unpacked under .cuda-toolkit-12.0 and exposed through the ignored
.cuda-home-12.0 view. Neither operation installed a system driver.

## PyTorch / 3D Stack

| Component | Version / value | Result |
|---|---|---|
| NumPy | 1.26.4 | OK |
| PyTorch | 2.5.1+cu124 | OK |
| torch.version.cuda | 12.4 | OK |
| torch.cuda.is_available() | True | PASS |
| GPU detected by Torch | NVIDIA GeForce RTX 2060 | PASS |
| Compute capability | 7.5 | PASS |
| cuDNN | 90100 (9.1.0.70) | PASS |
| torchvision | 0.20.1+cu124 | OK |
| spconv | 2.3.8 (spconv-cu124) | PASS |
| cumm | 0.7.11 (cumm-cu124) | OK |
| Numba | 0.60.0 | OK |
| llvmlite | 0.43.0 | OK |
| SciPy | 1.13.1 | OK |

PyTorch user-space CUDA libraries are 12.4. The driver capability (13.1),
Toolkit nvcc (12.0), and PyTorch build CUDA (12.4) are three different values.

## Compatibility Matrix

| Component | Current | Role | Status |
|---|---|---|---|
| Ubuntu | 24.04.4 LTS | Linux build environment | OK |
| WSL | 2.7.12.0 / WSL2 | GPU passthrough | OK |
| GPU | RTX 2060, sm_75 | CUDA extension target | OK |
| NVIDIA driver | 592.82 | Windows host driver | OK |
| Driver CUDA capability | 13.1 | Driver capability only | OK |
| Python | 3.12.3 | OpenPCDet Python 3.6+ | OK |
| PyTorch | 2.5.1+cu124 | OpenPCDet torch dependency | OK, tested |
| PyTorch CUDA | 12.4 | Torch runtime/build | OK |
| CUDA Toolkit | nvcc 12.0.140 | Extension compiler | OK, minor warning |
| GCC | 12.4 build compiler | CUDA host compiler | OK |
| spconv | 2.3.8 / cu124 | Sparse convolution | OK |
| OpenPCDet | commit below | Third-party backend | OK |

## OpenPCDet

| Component | Value |
|---|---|
| Repository | https://github.com/open-mmlab/OpenPCDet.git |
| Integration method | Git submodule at third_party/OpenPCDet |
| Commit | 233f849829b6ac19afb8af8837a0246890908755 |
| Package version | 0.6.0+233f849 |
| Install method | pip install -e . --no-build-isolation --no-deps |
| Source modified | No tracked source changes |
| Compiled modules | bev_pool, ingroup_inds, iou3d_nms, pointnet2_batch, pointnet2_stack, roiaware_pool3d, roipoint_pool3d |

The main repository tracks only the submodule gitlink. Generated version files,
caches, and compiled shared objects remain ignored inside the submodule.

## Installation Commands

The base image lacked python3.12-venv and root access was unavailable. The
actual isolated environment bootstrap was:

~~~bash
python3 -m venv --without-pip .venv
curl -L https://bootstrap.pypa.io/get-pip.py | .venv/bin/python -
.venv/bin/python -m pip install --upgrade pip setuptools wheel
~~~

On a normal Ubuntu install, sudo apt-get install python3.12-venv followed by
python3 -m venv .venv is preferred.

Torch and torchvision came from the official CUDA 12.4 wheels. SHA256:

~~~text
torch-2.5.1+cu124-cp312-cp312-linux_x86_64.whl
bf6484bfe5bc4f92a4a1a1bf553041505e19a911f717065330eb061afe0e14d7
torchvision-0.20.1+cu124-cp312-cp312-linux_x86_64.whl
d1053ec5054549e7dac2613b151bffe323f3c924939d296df4d7d34925aaf3ad
~~~

The exact remaining application dependencies were:

~~~bash
pip install numpy==1.26.4 llvmlite==0.43.0 numba==0.60.0
pip install tensorboardX==2.6.4 easydict==1.13 pyyaml==6.0.3
pip install scikit-image==0.24.0 scipy==1.13.1 pillow==10.4.0
pip install imageio==2.35.1 tifffile==2024.8.30 lazy-loader==0.4
pip install tqdm==4.67.1 SharedArray==3.2.4 opencv-python==4.10.0.84
pip install pyquaternion==0.9.9 protobuf==5.28.3
pip install spconv-cu124==2.3.8 cumm-cu124==0.7.11
pip install pccm==0.4.16 ccimport==0.4.4 pybind11==3.1.0
pip install fire==0.7.1 ninja==1.13.0 requests==2.34.2
pip install certifi==2026.7.22 charset-normalizer==3.5.1
pip install idna==3.19 urllib3==2.7.0 lark==1.3.1 portalocker==4.3.0
pip install termcolor==3.3.0
~~~

The PyTorch wheel pinned the CUDA runtime packages, Triton 3.1.0, and SymPy
1.13.1. The installed package inventory is recorded below.

No system CUDA or Linux NVIDIA driver was installed. User-space build artifacts:

~~~text
nvidia-cuda-toolkit Debian package 12.0.140
  SHA256 ee91ed0262cd2c66a24453d53d3bcfe60825abdd6536de997cd8fd8d4f50b483
nvidia-cuda-nvcc-cu12 12.4.131 (headers/ptxas support)
nvidia-cuda-cccl-cu12 12.4.127.post1 (Thrust/CUB/nv headers)
GCC/G++ 12.4.0 Debian packages
~~~

Submodule and build:

~~~bash
git submodule update --init --depth 1 third_party/OpenPCDet
git -C third_party/OpenPCDet checkout 233f849829b6ac19afb8af8837a0246890908755
cd third_party/OpenPCDet
CUDA_HOME=$PWD/../../.cuda-home-12.0 PATH=$PWD/../../.cuda-home-12.0/bin:$PWD/../../.gcc-12/usr/bin:$PATH CC=$PWD/../../.gcc-12/usr/bin/gcc-12 CXX=$PWD/../../.gcc-12/usr/bin/g++-12 TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS=1 NVCC_PREPEND_FLAGS=-I$PWD/../../.cuda-home-12.0/include python -m pip install -e . --no-build-isolation --no-deps
~~~

Modern setuptools initially attempted an isolated editable build and could not
see torch. The no-build-isolation form above is the equivalent validated
installation command; OpenPCDet setup.py was not patched.

## Package Inventory

Relevant validated pip freeze entries:

~~~text
Jinja2==3.1.6
MarkupSafe==3.0.3
PyYAML==6.0.3
SharedArray==3.2.4
ccimport==0.4.4
certifi==2026.7.22
charset-normalizer==3.5.1
cumm-cu124==0.7.11
easydict==1.13
filelock==3.32.4
fire==0.7.1
fsspec==2026.7.0
idna==3.19
imageio==2.35.1
lark==1.3.1
lazy_loader==0.4
llvmlite==0.43.0
mpmath==1.3.0
networkx==3.6.1
ninja==1.13.0
numba==0.60.0
numpy==1.26.4
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cccl-cu12==12.4.127.post1
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvcc-cu12==12.4.131
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
opencv-python==4.10.0.84
packaging==26.3
pccm==0.4.16
pillow==10.4.0
portalocker==4.3.0
protobuf==5.28.3
pybind11==3.1.0
pyquaternion==0.9.9
requests==2.34.2
scikit-image==0.24.0
scipy==1.13.1
setuptools==84.0.0
spconv-cu124==2.3.8
sympy==1.13.1
tensorboardX==2.6.4
termcolor==3.3.0
tifffile==2024.8.30
tqdm==4.67.1
torch==2.5.1+cu124
torchvision==0.20.1+cu124
triton==3.1.0
typing_extensions==4.16.0
urllib3==2.7.0
wheel==0.48.0
~~~

## Smoke Tests

| Test | Result | Command |
|---|---|---|
| WSL GPU passthrough | PASS | nvidia-smi |
| Torch CUDA availability | PASS | python -c "import torch; assert torch.cuda.is_available()" |
| Torch CUDA tensor | PASS | 1024x1024 CUDA matrix multiply and synchronize |
| spconv import | PASS | import spconv; import spconv.pytorch |
| pcdet import/source | PASS | import pcdet; print(pcdet.__file__) |
| iou3d_nms extension | PASS | boxes_iou_bev, nms_gpu |
| roiaware_pool3d extension | PASS | points_in_boxes_gpu |
| dependency consistency | PASS | PYTHONPATH= .venv/bin/python -m pip check |

The actual CUDA extension test returned:

~~~text
BEV IoU: [[0.6000000238418579, 0.0], [0.0, 0.0]]
NMS keep indices: [0, 1]
points_in_boxes: [[0, -1]]
~~~

This demonstrates Python -> PyTorch CUDA -> compiled OpenPCDet extension -> GPU.
pcdet import alone was not treated as an ops validation.

## Repository Hygiene

The main repository changes are only .gitmodules, the OpenPCDet submodule
reference, this lock document, the updated environment notes, and the
additional local-tool ignore rules. No datasets, checkpoints, private keys,
environment directories, or compiled shared objects are tracked.

## Known Limitations

- nvcc 12.0.140 and PyTorch CUDA 12.4 differ by a minor version. PyTorch emits
  a warning during extension builds, but all seven extensions and GPU tests
  passed. Keep this pair frozen unless a new compatibility investigation is
  performed.
- GCC 12.4 and the Toolkit are local unpacked build tools because this run had
  no root access. A clean machine must recreate those ignored directories or
  install equivalent system packages.
- No KITTI/nuScenes data, checkpoints, training, detector wrapper, geometry,
  evaluation, visualization, or other Phase 1+ business module was added.

## Phase 0 Status

PASS.

Repository is ready for Phase 1: KITTI Data & 3D Geometry.

## Phase 1 Tooling Additions

The Phase 0 core environment was not upgraded or downgraded. The following
Python packages were added for deterministic tests and the requested
visualization tools:

| Package | Version |
|---|---|
| pytest | 8.3.5 |
| matplotlib | 3.9.2 |
| open3d | 0.19.0 |

Open3D 0.19.0 has a Python 3.12 wheel in this environment. The package is
used only by the Phase 1 3D visualization path; NumPy/SciPy geometry remains
independent of it.

The project package itself is installed editable with:

~~~bash
python -m pip install -e . --no-deps --no-build-isolation
~~~

## Phase 2 Tooling Additions

The Phase 0 core versions remain unchanged. OpenPCDet's fixed revision imports
its optional Argoverse2 dataset module from `pcdet.datasets`; for this
revision, the following Python packages were added so the KITTI data/model
APIs can import under Python 3.12 and Torch 2.5.1:

| Package | Version | Purpose |
|---|---|---|
| kornia | 0.6.12 | Compatible Argo2 import dependency |
| av2 | 0.2.1 | Compatible Argo2 import dependency |
| gdown | 6.1.0 | Attempted official Model Zoo download |

Kornia 0.7.4 was tested first but its TorchScript conversion API failed to
compile under Torch 2.5.1; 0.6.12 imported successfully. These packages are
not used by the PointPillars network itself. The official checkpoint download
was attempted through Google Drive and was blocked by the current network
(`Network is unreachable`); no checkpoint was created.

OpenPCDet KITTI metadata was prepared without copying the raw dataset:

~~~bash
ln -sfn ~/datasets/kitti/training third_party/OpenPCDet/data/kitti/training
ln -sfn ~/datasets/kitti/testing third_party/OpenPCDet/data/kitti/testing
PYTHONPATH= .venv/bin/python -m pcdet.datasets.kitti.kitti_dataset \
  create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
~~~

This generated ignored `kitti_infos_train.pkl`, `kitti_infos_val.pkl`,
`kitti_infos_trainval.pkl`, `kitti_infos_test.pkl`, `kitti_dbinfos_train.pkl`,
and `gt_database/` under `third_party/OpenPCDet/data/kitti`.

## Phase 3 Tooling Additions

The Phase 0-2 torch/CUDA/spconv/OpenPCDet versions remain unchanged. The
nuScenes mini adapter and official evaluator use `nuscenes-devkit==1.2.0`,
which is compatible with the frozen Python 3.12 visualization stack. The
package was installed without dependency resolution; existing validated
OpenCV, NumPy, SciPy, Pillow, sklearn, and pyquaternion packages provide the
runtime dependencies:

| Package | Version |
|---|---|
| nuscenes-devkit | 1.2.0 |
| cachetools | 7.1.7 |
| descartes | 1.1.0 |
| shapely | 2.0.6 |
| pycocotools | 2.0.11 |
| torch-scatter | 2.1.2+pt25cu124 |

The available dataset is `~/datasets/nuscenes/v1.0-mini` (10 scenes, 404
samples). Raw data is linked into `third_party/OpenPCDet/data/nuscenes`; no
dataset files or generated infos are tracked. The official CenterPoint config
used for the validated Phase 3 run is
`cbgs_dyn_pp_centerpoint.yaml`; its matching Model Zoo checkpoint is recorded
in `configs/detectors/centerpoint/nuscenes_mini.yaml`. `torch-scatter` is
required by OpenPCDet's dynamic pillar VFE and is installed as the CUDA 12.4
wheel shown above.
