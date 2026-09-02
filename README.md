# LiDAR 三维感知

这是一个面向 KITTI 和 nuScenes 的 LiDAR 三维目标检测、评估与实验分析项目。项目负责数据适配、几何变换、`Box3D`/`PredictionBatch` schema、预测缓存 provenance、matching、评估、融合、验证工具和报告；CenterPoint 与 VoxelNeXt 网络来自固定 revision 的 OpenPCDet 预训练检测器。

默认模型是 **VoxelNeXt**，因为冻结的 `nuScenes v1.0-mini / mini_val` 对比中它具有最高的官方 mAP/NDS 和 precision。CenterPoint 用作 baseline；E3 是可选的预测结果 late-fusion 实验，能够改善远距离和稀疏目标的 custom recall，但会增加 FP，且官方 mAP/NDS 低于 VoxelNeXt，因此不是默认模型。

## 功能范围

- KITTI、nuScenes 数据适配和多 sweep 坐标变换。
- 统一的 `Box3D`、`PredictionBatch` 和 cache provenance。
- 基于中心距离的 class-aware one-to-one matching。
- 按距离、GT 点密度的诊断评估，官方 nuScenes 指标转换和 bootstrap。
- 冻结 E3 prediction-only fusion、环境/资产 validator 和单样本 demo。
- 可由 JSON artifacts 确定性生成的结果摘要。

## 快速开始：CPU 检查

这条路径不需要 GPU、nuScenes、checkpoint、PredictionCache 或 OpenPCDet checkout。使用 Python 3.12 安装轻量依赖和 CPU Torch：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[ci]'
.venv/bin/python -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

PYTHONPATH=. .venv/bin/python tools/validate_environment.py --profile cpu
PYTHONPATH=. .venv/bin/python tools/demo_nuscenes.py --help
make cpu-tests
PYTHONPATH=. .venv/bin/python tools/generate_phase6_summary.py --check
```

如果 `python3.12 -m venv` 报告 `ensurepip` 不可用，请先安装系统提供的 Python 3.12 venv 包。CPU suite 不会初始化 detector 或下载外部资产。

## GPU 单样本推理

真实 detector 推理还需要 CUDA 版 Torch、spconv、nuScenes 数据、匹配 SHA-256 的 checkpoint，以及已初始化并编译的 OpenPCDet：

```bash
git submodule update --init third_party/OpenPCDet
PYTHONPATH=. .venv/bin/python tools/validate_environment.py --profile gpu
PYTHONPATH=. .venv/bin/python tools/validate_assets.py --detector voxelnext
PYTHONPATH=. .venv/bin/python tools/demo_nuscenes.py --sample-token <token>
```

数据根目录可由 `NUSCENES_ROOT` 或 `--dataset-root <path>` 指定；单模型 checkpoint 可由 `--checkpoint <path>` 覆盖。默认输出为 `outputs/demo/<detector>/<sample-token>.json`，该目录不会进入 Git。

显式选择 CenterPoint 或 E3：

```bash
PYTHONPATH=. .venv/bin/python tools/demo_nuscenes.py --detector centerpoint --sample-token <token>
PYTHONPATH=. .venv/bin/python tools/demo_nuscenes.py --detector e3 --sample-token <token>
```

E3 按冻结配置顺序运行两个 detector，并直接融合内存中的 `PredictionBatch`；它不读取 GT，也不要求离线 cache。完整安装和兼容性说明见 [docs/quickstart.md](docs/quickstart.md) 与 [docs/environment.md](docs/environment.md)。

## 实验结果

下表来自已提交的 E3/E4 JSON artifacts。结果属于 **nuScenes v1.0-mini exploratory experiment**；`mini_val` 只有两个 scene，且此前已暴露，不代表 full nuScenes benchmark、full nuScenes 泛化或 SOTA claim。

| 方法 | 50m+ recall | 0–5 点 recall | precision | FP | mAP | NDS |
|---|---:|---:|---:|---:|---:|---:|
| CenterPoint | 14.00% | 63.30% | 21.76% | 12,841 | 0.4371 | 0.4919 |
| VoxelNeXt | 23.40% | 68.71% | 27.95% | 9,591 | 0.5209 | 0.5442 |
| E3 | 24.60% | 71.17% | 16.13% | 19,678 | 0.4996 | 0.5210 |

E3 的 recall 收益伴随 FP 增加和官方指标代价，所以保留为 directional ablation。完整四方法表、runtime 计时范围、图表和生成命令见 [reports/phase6_summary.md](reports/phase6_summary.md)。

## 项目结构与文档

- [docs/README.md](docs/README.md)：按功能组织的文档导航。
- [docs/architecture.md](docs/architecture.md)：模块职责、数据流、cache 和 fusion 边界。
- [docs/data_and_geometry.md](docs/data_and_geometry.md)：坐标系、box、点云和 sweep。
- [docs/evaluation.md](docs/evaluation.md)：指标定义、评估和 runtime 语义。
- [docs/third_party.md](docs/third_party.md)：第三方代码、模型、数据和许可证边界。
- [experiments/README.md](experiments/README.md)：E0–E4 原始实验索引；JSON、配置和 figures 保持不变。
- [docs/archive/README.md](docs/archive/README.md)：保留的历史协议和测量记录索引。

## 测试

无资产 CPU 检查使用 `make cpu-tests`；具备历史真实数据 fixture 时可运行完整套件：

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
```

GitHub Actions 在 Python 3.12 上运行同一组 focused CPU tests、环境检查和 summary 校验，不需要 CUDA 或 detector inference。

## 限制与后续工作

Phase 6 只覆盖此前暴露的 `nuScenes v1.0-mini` 两个 `mini_val` scene，不能代表 full nuScenes。Phase 5 detector E2E、E3 顺序总耗时估算和 demo cold/warm CLI wall time 的计时范围不同，不能直接比较。未随仓库分发 dataset、checkpoint、cache 或 runtime outputs。新 split/full nuScenes 验证、tracking 和 PyPI 发布属于后续工作。

## 许可证与第三方来源

本项目原创代码和文档采用 [Apache License 2.0](LICENSE)。OpenPCDet 保留其自身的 Apache-2.0 LICENSE 和版权声明；nuScenes、KITTI、预训练 checkpoint、spconv、Torch/CUDA 及其他第三方资产遵循各自来源条款，不由本项目许可证重新授权。OpenPCDet revision、checkpoint URL 和 SHA-256 见 [docs/third_party.md](docs/third_party.md) 与 `configs/detectors/`。
