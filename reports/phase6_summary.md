# Phase 6 结果摘要

本报告由已提交的 E3/E4 machine-readable artifacts 生成。运行
`PYTHONPATH=. .venv/bin/python tools/generate_phase6_summary.py --check` 可检查漂移。
结果仅属于 **nuScenes v1.0-mini exploratory experiment**：`mini_val` 有
81 个 sample、来自两个 scene，且此前已暴露；不代表 full nuScenes
benchmark、full nuScenes 泛化或 SOTA claim。

## mini_val 对比

Custom 指标使用 2 m 阈值的 class-aware one-to-one center matching。距离
recall 按 GT 中心距离分组；密度 recall 按 GT 框内当前 keyframe LiDAR 点数
分组。E2 的推理特征才是预测框内当前帧点数，不能与 GT 评估分组混淆。
官方列来自 nuScenes detection evaluator。

| 方法 | 50m+ recall | 0–5 点 recall | overall recall | precision | FP | mAP | NDS | runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CenterPoint | 14.00% | 63.30% | 80.41% | 21.76% | 12,841 | 0.4371 | 0.4919 | 70.24 ms* |
| VoxelNeXt | 23.40% | 68.71% | 83.79% | 27.95% | 9,591 | 0.5209 | 0.5442 | 111.69 ms* |
| Naive Union | 24.60% | 71.39% | 85.39% | 12.77% | 25,891 | 0.3900 | 0.4726 | N/A |
| E3 late fusion | 24.60% | 71.17% | 85.23% | 16.13% | 19,678 | 0.4996 | 0.5210 | 240.90 ms est. |

E3 是冻结的顺序 late-fusion directional ablation。它提高远距离和稀疏目标
recall，但 FP 增加、precision 降低；Naive Union 仅作高 recall/high-FP 对照。

## Runtime 语义

- CenterPoint 和 VoxelNeXt 是历史 Phase 5 detector E2E 测量：分别为
  70.24 ms/sample 和 111.69 ms/sample，
  包含预处理、传输、推理和 schema 转换。
- E3 的 **240.90 ms/sample** 是上述两项加缓存预测 CPU fusion 的
  顺序总耗时估算，不是重新测量的 detector E2E latency。
- VoxelNeXt demo 的约 785 s cold、41.4 s warm 是包含进程/模型启动
  和资产 I/O 的 CLI wall time，不能与 detector E2E 列混用。
- E4 repeat validation 状态为 **PASS**；它重复冻结 cache 评估，未重新调参。

## 推荐

默认使用 **VoxelNeXt**；CenterPoint 作为 baseline，E3 作为 directional
complementarity ablation，Naive Union 仅作诊断对照。

## 数据来源与图表

- [`experiments/e3_late_fusion/metrics.json`](../experiments/e3_late_fusion/metrics.json)
- [`experiments/e4_repeat_validation/metrics.json`](../experiments/e4_repeat_validation/metrics.json)
- [`experiments/e3_late_fusion/benchmark.json`](../experiments/e3_late_fusion/benchmark.json)

距离 recall 显示互补性主要集中在哪些范围，包括 E3 相对 VoxelNeXt 的小幅 50m+ 收益。

![Recall by distance](../experiments/e3_late_fusion/figures/recall_by_distance.png)

密度 recall 展示 0–5 点稀疏切片，避免它被 overall score 掩盖。

![Recall by point density](../experiments/e3_late_fusion/figures/recall_by_density.png)

官方 mAP/NDS 说明即使 E3 custom recall 增长，VoxelNeXt 仍是默认模型。

![Official mAP and NDS](../experiments/e3_late_fusion/figures/official_metrics.png)

互补性图区分共享检测和模型独有恢复结果，说明保留 E3 ablation 的原因。

![Detector complementarity](../experiments/e3_late_fusion/figures/complementarity.png)
