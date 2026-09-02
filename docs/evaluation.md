# 评估与实验结果

## 指标边界

matching 是 class-aware、one-to-one 的中心距离匹配，阈值为 `2.0 m`，
并使用 `distance <= threshold`。距离 recall 按 **GT 中心距离**分组；密度
recall 按 **GT 框内当前 keyframe LiDAR 点数**分组。E2 的推理特征才会数
**预测框内当前帧点数**，这两个概念不能互换。

自定义 recall、precision、FP、点数/距离切片是诊断指标，不是官方
nuScenes 指标。官方评估使用 nuScenes `detection_cvpr_2019`，报告 mAP、
NDS、mATE、mASE、mAOE、mAVE 和 mAAE。

## Phase 6 结果

完整四方法对比和图表见 [phase6_summary.md](../reports/phase6_summary.md)，
摘要由 `tools/generate_phase6_summary.py` 从 E3/E4 JSON 生成。实验是
`nuScenes v1.0-mini exploratory experiment`：`mini_val` 只有两个 scene，
此前已暴露，不代表 full nuScenes benchmark 或 SOTA。

| 方法 | 50m+ recall | 0–5 点 recall | precision | FP | mAP | NDS |
|---|---:|---:|---:|---:|---:|---:|
| CenterPoint | 14.00% | 63.30% | 21.76% | 12,841 | 0.4371 | 0.4919 |
| VoxelNeXt | 23.40% | 68.71% | 27.95% | 9,591 | 0.5209 | 0.5442 |
| E3 | 24.60% | 71.17% | 16.13% | 19,678 | 0.4996 | 0.5210 |

E3 保留为 directional ablation：custom recall 有提升，但 FP 增加且官方
mAP/NDS 低于 VoxelNeXt，因此默认模型仍是 VoxelNeXt。

## 复现与 runtime

E1–E4 的冻结 config、seed、split、cache provenance 和原始 figures 位于
`experiments/`。Phase 5 detector E2E、E3 顺序总耗时估算和 demo cold/warm
CLI wall time 计时范围不同，不能混列比较。
