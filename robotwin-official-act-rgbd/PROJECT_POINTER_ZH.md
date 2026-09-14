# RGB-D 模型项目重构入口

项目主线已经切换为“官方 ACT 动作核心 + 可替换深度视觉前端”。

- 新仓库：`/ssd/hhw/depth-model/repos/official-act-rgbd`
- 架构说明：`ARCHITECTURE_ZH.md`
- FairACT 迁移说明：`MIGRATION_FROM_FAIRACT_ZH.md`
- 训练与评测顺序：`RUN_PLAN_ZH.md`
- 机器可读实验矩阵：`experiment_matrix.json`
- 当前检查状态：`status.json`
- 完整预检：`/ssd/hhw/depth-model/manifests/official_act_rgbd_preflight.json`

旧 `/ssd/hhw/depth-model/repos/rgbd-act-v2` 保留作故障审计与历史结果，不再承担模型架构基准。
下一项主线工作是先在三个 sentinel tasks 上训练和评测 `ACT0_RGB`，通过官方行为复现门槛后再启动
`ACT1/ACT2/ACT4/ACT5` 深度架构筛选。

