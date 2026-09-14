# 从 FairACT 迁移到官方 ACT 基准

## 结论定位

当前 `RGBDACTv2/FairACT B0-B4` 的已完成结果仍然保留，但统一标记为“探索性结果”。它们可以说明
旧管线的行为和失败模式，不能用于比较深度架构优劣，因为旧 RGB 基线本身不是官方 ACT。

已发现的混杂因素包括：

- 输入由官方 640x480 改为 224x224，ResNet 特征从约 15x20 降为 7x7；
- 动作 CVAE、Transformer、位置编码和优化器由自定义实现替代；
- 预测 50 步却只执行前 6 步，随后丢弃剩余 44 步并重新规划；
- 训练改成逐帧 30,000 updates，不再是官方逐 episode/epoch 随机采样；
- checkpoint 只依据有序验证集前 50 个 batch，且看的是 posterior imitation loss；
- 深度反事实敏感度很低，旧模型大多依赖 RGB 与 joint。

因此，旧成功率低不能直接解释为“深度没用”，也不能据此选出 CNN、XYZ 或点云中的赢家。

## 可以继承的资产

- 六项 RoboTwin master dataset 和官方 ACT processed dataset；
- clean/noisy/processed depth sidecar 及无模型深度质量指标；
- 官方 ACT/DP3 的有效 seed 列表与 100-rollout 评测流程；
- SAPIEN 物理 GPU 隔离库和 GPU0 禁用约束；
- LingBot-Depth v0.5 官方仓库、权重、vendor 依赖与 `a3-eval` 环境；
- 旧模型视频、动作范围审计和 action-step diagnostic，作为故障分析材料。

## 不再继承的内容

- `FairACT` 的动作模型实现与 checkpoint；
- 224x224 方形输入、49 个固定 learned spatial embeddings；
- `action_steps=6`；
- frame-level 30k 训练预算和 EMA checkpoint；
- 仅凭 gate mean 或注意力热图判断深度有效性的做法。

## 架构映射

| 旧实验 | 新实验 | 是否可直接比较 |
| --- | --- | --- |
| `B0_RGB` | `ACT0_RGB` | 否，必须重新训练官方核心 |
| `B1_EARLY_RGBD` | `ACT1_EARLY_RGBD` | 否，只保留设计思想 |
| `B2_GATED_DEPTH` | `ACT2_DEPTH_CNN` | 否，新版只替换视觉前端 |
| `B3_GATED_XYZMAP` | `ACT4_XYZMAP` | 否，新版恢复官方分辨率与位置编码 |
| `B4_POINTCLOUD` | `ACT5_POINT_TOKENS` | 否；官方 DP3 另列外部参考 |

当前正在运行的 FairACT handover 评测允许自然结束，避免浪费已投入计算；不再从该队列扩展新的架构
结论。新仓库不会读取或覆盖旧 checkpoint。

