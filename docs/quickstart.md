# 快速开始

## CPU 检查

这条路径不需要 GPU、nuScenes、checkpoint 或 OpenPCDet 子模块。Python
3.12 环境只安装项目的 CPU 测试依赖和 CPU 版 Torch：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[ci]'
.venv/bin/python -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

PYTHONPATH=. .venv/bin/python tools/validate_environment.py --profile cpu
PYTHONPATH=. .venv/bin/python tools/demo_nuscenes.py --help
make cpu-tests
PYTHONPATH=. .venv/bin/python tools/generate_phase6_summary.py --check
```

如果系统提示 `ensurepip` 不可用，需要先安装发行版提供的 Python 3.12
venv 包。CPU validator、schema/geometry/evaluation 测试和报告检查不会
下载数据或初始化模型。

## GPU 单样本推理

真实 detector 推理需要已编译的 OpenPCDet、CUDA Torch、spconv、nuScenes
数据和匹配 SHA-256 的 checkpoint。先阅读
[environment.md](environment.md) 和 [environment.lock.md](environment.lock.md)，
再执行：

```bash
git submodule update --init third_party/OpenPCDet
PYTHONPATH=. .venv/bin/python tools/validate_environment.py --profile gpu
PYTHONPATH=. .venv/bin/python tools/validate_assets.py --detector voxelnext
PYTHONPATH=. .venv/bin/python tools/demo_nuscenes.py --sample-token <token>
```

数据根目录可通过 `NUSCENES_ROOT` 或 `--dataset-root <path>` 指定；单模型
checkpoint 可通过 `--checkpoint <path>` 覆盖。输出默认写入
`outputs/demo/<detector>/<sample-token>.json`，该目录被 Git 忽略。

默认模型是 VoxelNeXt。显式选择基线或 E3：

```bash
PYTHONPATH=. .venv/bin/python tools/demo_nuscenes.py --detector centerpoint --sample-token <token>
PYTHONPATH=. .venv/bin/python tools/demo_nuscenes.py --detector e3 --sample-token <token>
```

E3 按冻结配置顺序加载 CenterPoint 和 VoxelNeXt，直接融合两个
`PredictionBatch`；它不读取 GT，也不要求离线 PredictionCache。缓存是
Phase 6 离线实验的可选复现路径。
