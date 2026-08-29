# 《LiDAR 3D 感知与目标检测系统》项目需求与技术设计书 V1.0

> **项目代号**：LiDAR-3D-Perception  
> **目标用途**：秋招简历代表项目 / 3D Vision 学习主线 / 自动驾驶与机器人 3D 感知岗位作品集  
> **开发策略**：以 OpenPCDet 为第三方算法底座，围绕数据分析、几何工具、推理封装、分层评测、性能 Benchmark、稀疏目标优化与可视化进行自主二次开发  
> **主开发环境**：WSL2 + Ubuntu + NVIDIA CUDA  
> **文档版本**：V1.0  
> **状态**：立项 / 待 Phase 0 环境验收

---

## 0. 一页结论

本项目**不从 0 重写 3D Detection 框架**，也不做“Fork OpenPCDet + 改 README”的简单复现。

采用：

- **OpenPCDet**：第三方 3D Detection 底座
- **KITTI**：入门、调试、理解坐标系与 PointPillars
- **nuScenes**：主 Benchmark 与最终简历项目数据集
- **PointPillars**：教学 Baseline
- **CenterPoint**：主模型
- **VoxelNeXt**：高级对比模型
- **Open3D / 自研几何工具**：点云与 3D Geometry 能力补全
- **自研评测与分析系统**：Distance-aware / Point-density-aware Evaluation
- **核心研究问题**：远距离、低点云密度小目标检测
- **高级扩展**：3D Multi-Object Tracking；Camera-LiDAR 几何投影；有余力再考虑多模态融合

最终项目必须同时体现：

1. **3D Geometry 基础**
2. **Point Cloud 数据处理**
3. **现代 LiDAR 3D Detection**
4. **系统化评测与 Bad Case 分析**
5. **可复现优化实验**
6. **工程化与可视化**
7. **对第三方代码与本人贡献的清晰边界**

---

# 1. 项目背景与目标

## 1.1 背景

目标是构建一个不局限于单一工业场景的通用 3D Perception 项目，使项目能力能够覆盖以下岗位关键词：

- 3D 视觉算法
- LiDAR 感知
- 自动驾驶感知
- 机器人 3D 感知
- 点云算法
- BEV 感知
- 3D Detection
- 3D Tracking
- Camera-LiDAR 几何与多传感器基础

本项目**不是**为了覆盖所有 3D 方向。以下方向仅共享部分基础，不作为主线：

- SLAM / 定位建图
- SfM / MVS 三维重建
- NeRF / 3D Gaussian Splatting
- 工业 3D 测量 / 配准

## 1.2 最终项目名称

**中文：LiDAR 3D 感知与目标检测系统**

英文建议：

**LiDAR 3D Perception and Object Detection System**

如最终完成 Tracking，可升级为：

**LiDAR 3D Detection and Temporal Perception System**

## 1.3 应用场景

面向**自动驾驶与移动机器人道路环境感知**：

输入 360° LiDAR 点云，完成道路多类别目标的 3D 检测，输出：

- class
- confidence
- 3D center `(x, y, z)`
- size `(l, w, h)`
- yaw / orientation
- velocity（nuScenes / CenterPoint）
- distance
- point count in box

高级阶段增加：

- track_id
- trajectory
- temporal velocity smoothing

---

# 2. 项目成功标准

项目完成后，不以“能训练/能推理”为完成标准，而以以下能力闭环为验收：

## 2.1 算法闭环

必须能解释并验证：

- LiDAR 点云表示：`x, y, z, intensity`
- 坐标变换：LiDAR / Ego / Global / Camera
- Voxelization / Pillarization
- Sparse 3D Convolution
- BEV Feature
- 3D Bounding Box 参数化
- Center-based Detection
- 3D IoU / BEV IoU
- 3D NMS / Center-based post-processing
- nuScenes mAP / NDS 及主要 error metrics
- Distance / Density 与 Recall 的关系

## 2.2 工程闭环

必须具备：

- 一键数据分析
- 一键推理
- 一键官方评测
- 一键自定义分层评测
- 一键 Benchmark
- Scene 级 BEV / 3D 可视化
- 固定随机种子
- YAML 配置
- 日志
- 实验目录
- 结果自动汇总
- 测试
- 环境锁定
- 可复现 README

## 2.3 简历闭环

只有达到以下最低门槛后才能作为正式简历项目：

- [ ] PointPillars 跑通并理解
- [ ] CenterPoint 跑通并理解
- [ ] nuScenes 官方评测跑通
- [ ] 至少 2 个 Detector 完成统一 Benchmark
- [ ] Distance-aware Evaluation 完成
- [ ] Point-density-aware Evaluation 完成
- [ ] 至少 3 组优化消融
- [ ] 至少 1 个优化方案获得可复现正收益
- [ ] Scene 级 3D / BEV Demo 完成
- [ ] README、架构图、实验表格完整
- [ ] 所有简历数字均来自本人实际实验

Tracking 属于高级加分项，不允许为了 Tracking 拖死 Detection 主线。

---

# 3. 技术栈与第三方底座

## 3.1 核心技术栈

- Python
- PyTorch
- CUDA
- OpenPCDet
- spconv
- Open3D
- NumPy / SciPy
- nuScenes Devkit
- KITTI
- pytest
- YAML
- TensorBoard（可选）
- Docker（后期补充）
- Git / Git LFS

## 3.2 OpenPCDet 定位

OpenPCDet 只作为**第三方算法底座**，主要复用：

- Dataset 基础抽象
- KITTI / nuScenes Dataset loader
- 数据预处理与 augmentation 基础设施
- Voxel / Pillar 相关 ops
- PointPillars
- CenterPoint
- VoxelNeXt
- 训练框架
- 官方已有模型配置
- checkpoint 机制
- 3D ops / CUDA 扩展
- 官方基础 evaluation 接口

**禁止把 OpenPCDet 原始模块改名后当成自研。**

## 3.3 数据集定位

### KITTI

用途：

- Phase 1～2 学习
- PointPillars 入门
- LiDAR / Camera Calibration 学习
- 3D Box 与投影验证
- 小规模快速调试

不是最终主 Benchmark。

### nuScenes

用途：

- 主数据集
- CenterPoint 主模型
- VoxelNeXt 对比
- 多 Sweep
- 10 类道路目标
- velocity
- NDS / mAP
- Distance / Density 分析
- Scene 时序可视化
- Tracking 扩展

---

# 4. 开发环境决策

## 4.1 最终决策

**推荐：Windows 主机 + WSL2 Ubuntu 作为主开发/训练环境。**

不推荐把 OpenPCDet 主工程直接开发在原生 Windows Python 环境中。

## 4.2 原因

OpenPCDet 官方安装文档以 Linux 环境为主要测试环境，项目依赖 CUDA 扩展、spconv 等 3D 稀疏卷积组件，Linux 生态通常更稳定。

WSL2 可以：

- 使用 Linux 编译链
- 使用 NVIDIA GPU CUDA
- 保留 Windows 桌面体验
- 使用 VS Code / Terminal / Codex 等编辑工具
- 后期更容易迁移到服务器或 Linux 工作站
- 更接近真实 AI / 自动驾驶研发环境

## 4.3 推荐形态

```text
Windows 11
│
├── NVIDIA Windows Driver
│
├── VS Code / Terminal
│
└── WSL2
    └── Ubuntu
        ├── Git
        ├── Conda/Mamba
        ├── PyTorch
        ├── CUDA user-space toolkit（按需）
        ├── spconv
        ├── OpenPCDet
        └── lidar-3d-perception
```

## 4.4 重要限制

- **不要在 WSL 内安装 Linux NVIDIA 显卡驱动。**
- NVIDIA GPU 驱动装在 Windows 主机，WSL 使用映射的 CUDA driver。
- CUDA Toolkit、PyTorch、spconv 版本必须在 Phase 0 做兼容性冻结。
- 不在 V1.0 里硬编码一个未经本机验证的 PyTorch/CUDA 组合。

## 4.5 文件存储原则

**项目代码和训练数据尽量放在 WSL Linux 文件系统内**，例如：

```text
~/workspace/lidar-3d-perception
~/datasets/kitti
~/datasets/nuscenes
```

不建议长期把训练数据放在：

```text
/mnt/c/...
/mnt/d/...
```

原因是大量小文件 / 数据加载场景下跨 Windows 文件系统访问可能影响 IO 与权限体验。

大数据集如果因磁盘容量必须放 Windows 分区，可在 Phase 0 实测 dataloader 吞吐后再决定。

---

# 5. 总体系统架构

```text
                         ┌───────────────────────┐
                         │  KITTI / nuScenes     │
                         └──────────┬────────────┘
                                    │
                          Dataset Adapter
                                    │
                    ┌───────────────▼───────────────┐
                    │  Point Cloud Preprocessing    │
                    │  Range / Sweep / Aug / Voxel │
                    └───────────────┬───────────────┘
                                    │
               ┌────────────────────▼────────────────────┐
               │           3D Detector Backend           │
               │ PointPillars / CenterPoint / VoxelNeXt │
               └────────────────────┬────────────────────┘
                                    │
                         3D Boxes / Velocity
                                    │
                 ┌──────────────────▼──────────────────┐
                 │      Unified Prediction Schema     │
                 └───────┬──────────┬──────────┬───────┘
                         │          │          │
                         ▼          ▼          ▼
                    Evaluation  Analysis  Visualization
                         │          │          │
                         │          │          └─ BEV / 3D / Scene
                         │          │
                         │          ├─ Distance-aware
                         │          ├─ Density-aware
                         │          └─ Bad Case Mining
                         │
                         ├─ Official Metrics
                         └─ Runtime / VRAM Benchmark

高级扩展：
Unified Predictions → 3D Tracker → Track ID / Trajectory → Tracking Eval
```

---

# 6. 仓库目录设计

```text
lidar-3d-perception/
│
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml / requirements/
├── Makefile
│
├── configs/
│   ├── system/
│   ├── datasets/
│   ├── detectors/
│   │   ├── pointpillar/
│   │   ├── centerpoint/
│   │   └── voxelnext/
│   ├── tracker/
│   └── experiments/
│
├── lidar_perception/
│   ├── __init__.py
│   │
│   ├── datasets/
│   │   ├── schemas.py
│   │   ├── kitti_adapter.py
│   │   └── nuscenes_adapter.py
│   │
│   ├── geometry/
│   │   ├── transforms.py
│   │   ├── boxes3d.py
│   │   ├── projection.py
│   │   ├── pointcloud.py
│   │   └── registration.py
│   │
│   ├── detection/
│   │   ├── base.py
│   │   ├── openpcdet_backend.py
│   │   └── schemas.py
│   │
│   ├── evaluation/
│   │   ├── official.py
│   │   ├── distance_eval.py
│   │   ├── density_eval.py
│   │   ├── matching.py
│   │   └── metrics.py
│   │
│   ├── analysis/
│   │   ├── dataset_stats.py
│   │   ├── point_density.py
│   │   ├── distance_stats.py
│   │   ├── badcase_mining.py
│   │   └── report.py
│   │
│   ├── benchmark/
│   │   ├── latency.py
│   │   ├── memory.py
│   │   └── throughput.py
│   │
│   ├── visualization/
│   │   ├── bev.py
│   │   ├── pointcloud_3d.py
│   │   ├── camera_projection.py
│   │   └── scene_player.py
│   │
│   ├── tracking/                 # P1 高级扩展
│   │   ├── kalman.py
│   │   ├── association.py
│   │   ├── tracker.py
│   │   └── schemas.py
│   │
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       ├── seed.py
│       └── io.py
│
├── tools/
│   ├── prepare_data.py
│   ├── analyze_dataset.py
│   ├── infer.py
│   ├── evaluate.py
│   ├── benchmark.py
│   ├── visualize.py
│   ├── visualize_scene.py
│   └── track.py
│
├── experiments/
│   ├── pointpillar_baseline/
│   ├── centerpoint_baseline/
│   ├── voxelnext_baseline/
│   ├── distance_ablation/
│   ├── density_ablation/
│   └── optimization/
│
├── outputs/
│   ├── predictions/
│   ├── metrics/
│   ├── figures/
│   └── demos/
│
├── docs/
│   ├── 00_environment.md
│   ├── 01_point_cloud_basics.md
│   ├── 02_coordinate_systems.md
│   ├── 03_voxel_and_pillar.md
│   ├── 04_pointpillars.md
│   ├── 05_sparse_conv.md
│   ├── 06_centerpoint.md
│   ├── 07_nuscenes_metrics.md
│   ├── 08_distance_density_analysis.md
│   ├── 09_tracking.md
│   └── 10_experiment_log.md
│
├── tests/
│   ├── test_geometry.py
│   ├── test_boxes3d.py
│   ├── test_projection.py
│   ├── test_matching.py
│   └── test_schemas.py
│
└── third_party/
    └── OpenPCDet/                # git submodule 或固定 commit
```

---

# 7. OpenPCDet 与自研边界

## 7.1 允许直接复用 OpenPCDet

| 模块 | 处理方式 | 简历归属 |
|---|---|---|
| PointPillars 网络主体 | 直接复用/配置 | 第三方 Baseline |
| CenterPoint 网络主体 | 直接复用/配置 | 第三方 Baseline |
| VoxelNeXt 网络主体 | 直接复用/配置 | 第三方 Baseline |
| KITTI loader | 复用 | 第三方基础设施 |
| nuScenes loader | 复用 | 第三方基础设施 |
| voxel CUDA ops | 复用 | 第三方基础设施 |
| spconv | 复用 | 第三方依赖 |
| train loop | 复用 | 第三方基础设施 |
| checkpoint | 复用 | 第三方基础设施 |
| 原始 augmentation | 复用后可扩展 | 第三方 + 自研改动 |
| 官方 evaluation adapter | 复用 | 第三方基础 |

## 7.2 必须由本项目自主实现

| 模块 | 必须完成的自主工作 |
|---|---|
| 项目统一配置层 | 将 OpenPCDet backend 与自研模块解耦 |
| Unified Prediction Schema | 统一不同 detector 输出 |
| Dataset Analysis | 类别/距离/点数/Box 尺寸统计 |
| 3D Geometry Toolkit | 坐标变换、3D Box、投影等核心工具 |
| Distance-aware Evaluation | 自定义距离分层 |
| Point-density-aware Evaluation | 按 GT Box 点数分层 |
| Bad Case Mining | 自动提取 FP/FN/定位差样本 |
| Benchmark | latency / FPS / VRAM / throughput |
| Visualization | 统一 BEV / 3D / Scene 可视化 |
| Experiment Report | 自动生成 CSV/JSON/Markdown 汇总 |
| 优化实验 | 数据/特征/训练至少三个方向 |
| Tests | 自研代码测试 |
| README / Docs | 完整技术说明与学习总结 |

## 7.3 允许二次修改但必须记录 diff

若修改 OpenPCDet 内部实现，例如：

- 新增 distance feature
- 修改 VFE 输入
- 修改 loss weighting
- 修改 sampling strategy
- 自定义 head / augmentation

必须：

1. 在独立 branch 完成
2. 保留 commit
3. 在 `docs/10_experiment_log.md` 写明：
   - 原始实现
   - 修改点
   - 修改原因
   - 实验结果
4. 简历只描述本人真实修改内容

---

# 8. 核心数据结构 / 接口

## 8.1 PointCloudFrame

```python
@dataclass
class PointCloudFrame:
    frame_id: str
    points: np.ndarray        # [N, 4+] x,y,z,intensity,...
    timestamp: int | float
    lidar_to_ego: np.ndarray  # [4,4]
    ego_to_global: np.ndarray # [4,4]
    metadata: dict
```

## 8.2 Box3D

```python
@dataclass
class Box3D:
    center: np.ndarray        # [3]
    size: np.ndarray          # [3], l/w/h 需统一定义
    yaw: float
    label: str
    score: float | None
    velocity: np.ndarray | None
    track_id: str | None
```

## 8.3 PredictionBatch

```python
@dataclass
class PredictionBatch:
    frame_id: str
    boxes: list[Box3D]
    runtime_ms: float | None
    metadata: dict
```

## 8.4 DetectorBackend

```python
class DetectorBackend(ABC):
    @abstractmethod
    def load(self, config_path, checkpoint_path): ...

    @abstractmethod
    def predict(self, frame: PointCloudFrame) -> PredictionBatch: ...

    @abstractmethod
    def name(self) -> str: ...
```

OpenPCDet 必须被封装在 `OpenPCDetBackend` 内，避免整个自研系统与第三方框架强耦合。

## 8.5 Evaluator

```python
class Evaluator(ABC):
    def evaluate(
        self,
        predictions: list[PredictionBatch],
        ground_truth,
    ) -> dict:
        ...
```

实现：

- `OfficialEvaluator`
- `DistanceAwareEvaluator`
- `DensityAwareEvaluator`

---

# 9. 自研 3D Geometry Toolkit

这一模块是项目“通用 3D 能力”的关键，不能全部依赖 OpenPCDet 黑盒。

## 9.1 必须实现

- homogeneous transform
- rotation / translation
- quaternion ↔ rotation matrix
- transform points
- transform 3D boxes
- 3D box corners
- BEV corners
- point-in-box
- count points in box
- LiDAR → Camera projection
- 3D Box → Image projection

## 9.2 Open3D 辅助模块

至少实践：

- voxel downsampling
- statistical outlier removal
- KD-Tree
- normal estimation
- RANSAC plane segmentation
- ICP registration

上述算法**不要求全部接进 Detector 主链路**，但应作为 `geometry/` 中的独立 Demo / Tool，保证项目不仅是“深度学习调包”。

---

# 10. 评测体系

## 10.1 官方评测

KITTI：

- BEV AP
- 3D AP
- Easy / Moderate / Hard

nuScenes：

- mAP
- NDS
- mATE
- mASE
- mAOE
- mAVE
- mAAE

## 10.2 Distance-aware Evaluation

初版建议分桶：

```text
0–10 m
10–20 m
20–30 m
30–40 m
40–50 m
50 m+
```

输出至少：

- GT count
- prediction count
- recall
- precision
- matched localization error

最终分桶范围允许按数据分布调整，但必须在报告中记录。

## 10.3 Point-density-aware Evaluation

根据每个 GT 3D Box 内 LiDAR 点数分桶：

```text
0–5
6–10
11–20
21–50
51+
```

输出：

- GT count
- matched count
- recall
- average confidence
- localization error

核心目的：验证

```text
distance ↑
→ point density ↓
→ feature quality ↓
→ small-object miss rate ↑
```

## 10.4 Bad Case Mining

自动导出：

- False Negative
- False Positive
- low confidence TP
- high localization error TP
- far-range miss
- low-density miss

每类 Bad Case 保存：

- frame id
- class
- distance
- point count
- GT box
- prediction box
- confidence
- BEV snapshot / 3D snapshot

---

# 11. 性能 Benchmark

统一 Benchmark 必须固定：

- GPU
- CPU
- CUDA
- PyTorch
- batch size
- warmup count
- test iterations
- input data source
- precision FP32/FP16
- 是否包含 preprocessing / postprocessing

输出：

| Model | mAP | NDS | Latency | FPS | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| PointPillars | 实测 | 实测 | 实测 | 实测 | 实测 |
| CenterPoint | 实测 | 实测 | 实测 | 实测 | 实测 |
| VoxelNeXt | 实测 | 实测 | 实测 | 实测 | 实测 |
| Optimized | 实测 | 实测 | 实测 | 实测 | 实测 |

**禁止引用论文数字冒充本机结果。**

---

# 12. 核心研究问题与优化路线

## 12.1 核心问题

**远距离、低点云密度小目标检测**

重点类别：

- pedestrian
- bicycle
- motorcycle
- traffic_cone

## 12.2 Level A：数据层优化

候选实验：

- GT Sampling
- class-balanced sampling
- distance-aware sampling
- small-object oversampling
- object-level augmentation
- point dropout / sparsity simulation

## 12.3 Level B：特征层优化

候选实验：

- 显式 range feature：`r = sqrt(x^2 + y^2)`
- local point density encoding
- point count statistics
- distance-conditioned feature
- VFE 输入特征消融

所有新增特征必须做 ablation：

```text
baseline
+ range
+ density
+ range+density
```

## 12.4 Level C：训练层优化

候选实验：

- distance-aware loss weighting
- class × distance weighting
- hard sparse-object mining
- confidence calibration（可选）

注意：

**方案是否进入最终简历，只看可复现实验结果，不看“听起来高级”。**

---

# 13. 3D Tracking 高级扩展

优先级：P1

Detection 主线完成后再做。

最小实现：

```text
3D Detection
→ motion prediction
→ data association
→ Kalman Filter
→ track lifecycle
→ track_id
```

建议自主实现：

- Kalman state
- Hungarian association
- distance / 3D overlap cost
- track birth
- track confirmation
- lost/dead state

可参考公开 3D MOT 思路，但必须重新封装并理解。

评测：

- nuScenes Tracking Benchmark
- AMOTA 等官方指标
- ID switch / track length 辅助分析

---

# 14. Camera-LiDAR 几何扩展

优先级：P1/P2

**先做几何投影，不急于直接上 BEVFusion。**

必须完成：

```text
LiDAR Point
→ LiDAR frame
→ Ego frame
→ Camera frame
→ Camera intrinsic
→ image pixel
```

实现：

- LiDAR points overlay on image
- 3D box projection on image
- depth color visualization

这一模块用于证明多传感器空间几何能力。

---

# 15. 阶段开发计划与验收

## Phase 0：环境与仓库初始化

### Codex 任务

- 检测 WSL / Ubuntu / GPU / Driver
- 建立环境兼容矩阵
- 安装并固定 OpenPCDet 可运行环境
- 编译 CUDA ops
- 运行最小 smoke test
- 建立主仓库与 third_party 结构
- 固定 OpenPCDet commit
- 输出 `environment.lock.md`

### 验收

- [ ] `nvidia-smi` 在 WSL 可用
- [ ] `torch.cuda.is_available() == True`
- [ ] spconv import 成功
- [ ] pcdet import 成功
- [ ] CUDA ops smoke test 成功
- [ ] 环境版本全部记录
- [ ] Git commit 固定

**Phase 0 不写业务功能。**

---

## Phase 1：KITTI 数据与 3D Geometry

### Codex 任务

- KITTI adapter
- PointCloudFrame
- Box3D
- calibration parser
- coordinate transforms
- point-in-box
- points count
- LiDAR → image projection
- GT BEV / 3D visualization
- dataset statistics

### 学习重点

- 一帧 LiDAR 点云是什么
- KITTI 坐标系
- Calibration
- 3D Box
- homogeneous transform
- projection

### 验收

- [ ] 可加载指定 frame
- [ ] 可显示点云
- [ ] 可显示 GT 3D Box
- [ ] 可投影到 image
- [ ] 几何 unit test 全部通过
- [ ] 能解释每个坐标系

---

## Phase 2：PointPillars Baseline

### Codex 任务

- OpenPCDetBackend
- pretrained PointPillars inference
- Unified Prediction Schema
- Prediction → BEV/3D visualization
- KITTI evaluation wrapper
- baseline benchmark

### 学习重点

- Pillarization
- PFN / VFE
- pseudo image
- 2D backbone
- anchor
- target assignment
- NMS

### 验收

- [ ] 可一键推理
- [ ] 可显示 GT vs Pred
- [ ] KITTI eval 成功
- [ ] baseline latency 可复现
- [ ] 能独立画 PointPillars pipeline

---

## Phase 3：nuScenes + CenterPoint 主线

### Codex 任务

- nuScenes adapter
- multi-sweep 数据接入
- CenterPoint config / checkpoint
- official nuScenes evaluation
- scene visualization
- velocity visualization

### 学习重点

- nuScenes sensor / ego / global frame
- sweep
- voxel
- sparse convolution
- BEV
- heatmap
- center-based detector
- velocity regression
- NDS

### 验收

- [ ] nuScenes 数据准备完成
- [ ] CenterPoint inference 成功
- [ ] official mAP/NDS 成功
- [ ] Scene-level Demo 成功
- [ ] 能解释 CenterHead 的主要输出

---

## Phase 4：自研分层评测与 Bad Case

### Codex 任务

- DistanceAwareEvaluator
- DensityAwareEvaluator
- matching
- badcase_mining
- CSV/JSON report
- figure generation

### 验收

- [ ] 距离分桶结果正确
- [ ] 点数分桶结果正确
- [ ] 人工抽样验证 matching
- [ ] 自动导出典型 Bad Cases
- [ ] 得到第一版问题结论

这一步是项目从“复现”升级为“研究型工程项目”的关键。

---

## Phase 5：多模型 Benchmark

### Codex 任务

- PointPillars / CenterPoint / VoxelNeXt 统一 backend
- warmup / sync / latency
- peak VRAM
- benchmark report

### 验收

- [ ] 至少 2 个模型
- [ ] 最好 3 个模型
- [ ] 相同硬件、相同测试协议
- [ ] mAP/NDS/FPS/VRAM 同表
- [ ] README 可复现实验命令

---

## Phase 6：远距离稀疏目标优化

### Codex 任务

每次只做一个实验分支。

建议顺序：

1. sampling / augmentation
2. range feature
3. density feature
4. loss weighting
5. combination

### 每个实验必须输出

```text
Hypothesis
Change
Config diff
Training setup
Metrics
Distance metrics
Density metrics
Runtime
Conclusion
```

### 验收

- [ ] 至少 3 组独立消融
- [ ] 至少一个方案正收益
- [ ] 结果重复验证
- [ ] 不允许 cherry-pick 单次最好结果
- [ ] 最终保留方案有明确因果解释

---

## Phase 7：工程化与作品集封装

### Codex 任务

- README
- architecture diagram source
- demo script
- Docker（可选但推荐）
- tests
- Makefile
- clean setup guide
- results table
- GIF / MP4 export
- experiment index

### 验收

陌生开发者按照 README：

1. 安装
2. 准备 sample
3. 下载 checkpoint
4. 运行 demo
5. 看到输出

不得依赖作者本地隐藏路径。

---

## Phase 8：3D Tracking（高级）

只有 Phase 0～7 完成后进入。

### 验收

- [ ] Kalman
- [ ] association
- [ ] lifecycle
- [ ] scene tracks
- [ ] official / 辅助 tracking metrics

---

# 16. Codex 协作规则

## 16.1 禁止“一次性生成整个项目”

Codex 每次只接受一个 Phase 或 Phase 内一个子任务。

推荐工作流：

```text
设计书
→ Phase Prompt
→ Codex 实现
→ 单元测试 / Smoke Test
→ 人工 Review
→ 学习该模块
→ Git Commit
→ 下一 Phase
```

## 16.2 每次 Codex 开发前必须输出

1. 本阶段目标
2. 将修改/新增哪些文件
3. 依赖哪些已有接口
4. 不会修改哪些第三方代码
5. 风险
6. 验收命令

## 16.3 每次 Codex 开发后必须输出

1. 文件变更清单
2. 核心设计
3. 运行命令
4. 测试结果
5. 已知限制
6. 本阶段需要人工学习的 5～10 个知识点
7. 下一阶段建议

## 16.4 Git 要求

建议 branch：

```text
main
develop
phase/01-geometry
phase/02-pointpillars
phase/03-centerpoint
exp/range-feature
exp/density-feature
exp/distance-loss
```

每个重要实验独立 commit / branch。

---

# 17. 实验记录规范

每个实验目录：

```text
experiments/<exp_name>/
├── config.yaml
├── command.sh
├── environment.txt
├── metrics.json
├── benchmark.json
├── analysis.md
└── figures/
```

`analysis.md` 模板：

```markdown
# Experiment

## Hypothesis

## Baseline

## Change

## Controlled Variables

## Main Metrics

## Distance-aware Metrics

## Density-aware Metrics

## Runtime

## Result

## Failure Cases

## Conclusion

## Next Experiment
```

---

# 18. README 最终必须包含

1. 项目简介
2. 真实 Demo
3. 系统架构图
4. Why this project
5. Dataset
6. Model Backends
7. 自研模块
8. Quick Start
9. Evaluation
10. Benchmark
11. Distance / Density Analysis
12. Optimization Ablation
13. Bad Cases
14. Tracking（若完成）
15. 目录结构
16. Third-party Attribution
17. Reproducibility
18. Learning Notes

---

# 19. 最终简历目标模板

> **注意：以下所有数字均为占位符，必须以实际实验替换。**

### LiDAR 3D 感知与目标检测系统

**技术栈：** Python / PyTorch / OpenPCDet / CUDA / spconv / Open3D / nuScenes / CenterPoint / VoxelNeXt

- 基于 OpenPCDet 与 nuScenes 构建 LiDAR 3D 感知 Pipeline，完成多 Sweep 点云处理、Voxel/Sparse 表征、CenterPoint/VoxelNeXt 检测、3D/BEV Scene 可视化及官方 mAP/NDS 评测，支持道路多类别 3D Box、朝向与速度估计。
- 自主构建 **Distance-aware 与 Point-density-aware 分层评测及 Bad Case Mining**，量化目标距离、Box 内点数与 Recall/定位误差关系，定位远距离低密度 Pedestrian/Bicycle 等目标的主要漏检模式。
- 针对远距离稀疏目标，从数据采样、几何特征与训练损失开展消融实验，通过 **XXX** 将 **XXX 指标从 A 提升至 B**，并在统一硬件下完成 PointPillars/CenterPoint/VoxelNeXt 的 mAP/NDS/FPS/VRAM Benchmark。
- （完成后可选）扩展 3D MOT，基于 Kalman Filter 与数据关联维护时序轨迹，并完成 scene-level tracking evaluation。

---

# 20. 面试能力验收

项目最终不仅要求“会跑”，还要求能够回答：

### 3D Geometry

- LiDAR 坐标系和 Camera 坐标系区别？
- 4×4 齐次变换矩阵是什么？
- 为什么要用 quaternion？
- LiDAR 点怎么投影到图像？
- 3D Box 的 yaw 在哪个坐标系定义？

### Point Cloud

- 点云为什么稀疏？
- 为什么远距离目标更难？
- Voxel 与 Pillar 有什么区别？
- voxel size 对精度、速度、显存有什么影响？
- Sparse Convolution 为什么适合点云？

### Detection

- PointPillars 如何把 3D 转成 pseudo image？
- CenterPoint 为什么是 anchor-free？
- Heatmap target 怎么生成？
- CenterPoint 预测哪些量？
- NMS / center-distance post-process 如何工作？

### Evaluation

- KITTI AP 和 nuScenes mAP 有什么区别？
- NDS 是什么？
- mATE / mAOE / mAVE 分别是什么？
- 为什么仅看 overall mAP 不够？
- Distance / Density 分层评测为什么有价值？

### Experiment

- 最有效的优化是什么？
- 为什么有效？
- 哪些方案失败了？
- 是否存在速度/显存代价？
- 如何排除偶然波动？

---

# 21. 项目边界 / Non-Goals

V1.x 不主动加入：

- 完整 SLAM
- NeRF
- 3DGS
- SfM/MVS
- 端到端自动驾驶规划
- 大型多模态 Foundation Model
- 全套 Camera BEV
- 全套 Radar Fusion

原因：

**高级 ≠ 堆功能。**

本项目主线必须始终保持：

```text
3D Geometry
+ LiDAR Point Cloud
+ Modern 3D Detection
+ Evaluation / Bad Case
+ Optimization
+ Engineering
```

---

# 22. 风险与应对

## 22.1 OpenPCDet 环境年代较旧

风险：

- PyTorch / CUDA / spconv API 兼容问题
- CUDA ops 编译问题

策略：

- Phase 0 先做 smoke test
- 固定 OpenPCDet commit
- 固定成功环境
- 输出 environment lock
- 不在开发中随意升级 torch / CUDA

## 22.2 nuScenes 数据量与训练成本

策略：

- KITTI 做早期调试
- nuScenes mini 做 pipeline smoke test
- full train/val 用于正式实验
- 先 pretrained inference，再考虑训练
- 根据 GPU 显存调整 batch，不擅自改变评测协议

## 22.3 WSL IO

策略：

- 优先 WSL ext4 文件系统
- 数据量过大时实测 `/mnt/*` 与 WSL 文件系统 dataloader 吞吐
- 根据硬盘容量做取舍

## 22.4 项目过度膨胀

策略：

- Phase 0～7 为主线
- Tracking 为 P1
- Camera-LiDAR Fusion 为 P1/P2
- 不同时并行开发多个高级模块

---

# 23. Phase 0 交给 Codex 时的第一条原则

新聊天窗口中，先把本设计书完整提供给 Codex，然后要求：

> **不要直接编码。先阅读设计书，复述项目目标、OpenPCDet 与自研边界、阶段顺序和当前 Phase 0 验收标准；之后只为 Phase 0 生成详细开发 Prompt/执行计划，未经确认不得进入 Phase 1。**

这样可以避免 Codex 一上来生成大量不可控代码。

---

# 24. 参考项目与文档

- OpenPCDet: https://github.com/open-mmlab/OpenPCDet
- OpenPCDet Installation: https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md
- OpenPCDet Getting Started: https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md
- OpenPCDet Demo: https://github.com/open-mmlab/OpenPCDet/blob/master/docs/DEMO.md
- NVIDIA CUDA on WSL: https://docs.nvidia.com/cuda/wsl-user-guide/
- nuScenes: https://www.nuscenes.org/
- Open3D: https://github.com/isl-org/Open3D

---

# 25. V1.0 最终冻结项

以下内容在开始开发前视为已确认：

- **主开发平台：WSL2 Ubuntu**
- **第三方 3D Detection 底座：OpenPCDet**
- **入门数据集：KITTI**
- **主数据集：nuScenes**
- **入门 Baseline：PointPillars**
- **主模型：CenterPoint**
- **高级对比：VoxelNeXt**
- **核心自主模块：Geometry / Evaluation / Analysis / Benchmark / Visualization**
- **核心研究问题：远距离低点云密度小目标检测**
- **高级扩展：3D Tracking**
- **Camera-LiDAR：先做几何投影，融合模型后置**
- **所有指标必须真实实验**
- **不将第三方原有模块冒充自研**
- **按 Phase 分阶段交给 Codex，不允许一次性开发全部内容**

---

## 结语

这个项目的最终目标不是“会使用 OpenPCDet”，而是：

> **以 OpenPCDet 为可靠算法底座，自主搭建一套具有 3D Geometry、LiDAR 点云处理、现代 3D Detection、系统评测、Bad Case 分析、优化实验与工程可复现能力的完整 3D Perception 项目。**

当 Phase 0～7 全部通过后，该项目才进入“简历可用”状态。
