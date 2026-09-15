# 后续训练与评测计划

> 本文件保留架构筛选细节。跨机器执行、硬件准入、产物校验和严格状态机以
> `EXECUTION_PLAN_ZH.md` 和 `execution_plan.json` 为唯一准则。

## Q0：八架构官方6000轮训练与评测

只在 `stack_blocks_two/depth_master_clean` 上训练 ACT0-ACT7 八种架构。全部使用官方 6000 epochs、
batch 8、seed 0、逐 epoch 随机验证和同一数据划分。训练全部完成后立即以官方 temporal aggregation
在同一100-seed列表上评测。ACT0 是官方 RGB + joint；其余七种只使用 clean metric depth。

若差值达到 10 个百分点，继续做 2000 epochs + 100 episodes 确认；否则按唯一计划中的规则扩大
样本或排查管线。Q0 结果不能直接作为最终深度有效性结论。

## Q1：同任务完整确认

Q0 后只对 `stack_blocks_two` 的 `ACT0_RGB` 与 `ACT1_EARLY_RGBD` 延长到官方 2000 epochs，
评测使用同一份 100-seed 列表。多任务扩展放到架构方向得到确认之后。

代码 parity 已经要求模型初始化逐参数一致。既有官方复现结果 stack 35%、handover 89%、clutter
5% 作为历史参照。受控版修复为训练和部署都使用 `head-left-right`，而旧官方流程训练配置为
`head-right-left`、部署代码为 `head-left-right`，所以不要求逐百分点复现。若受控 `ACT0` 明显退化，
则暂停深度训练并先查相机、动作 chunk、控制器异常和 checkpoint 选择；若提升则正常保留。

## P1：低成本架构筛选

完成 Q0 后，重点比较：

1. `ACT2_DUAL_SHARED`
2. `ACT3_DUAL_PER_VIEW`
3. `ACT4_XYZMAP`
4. `ACT5_POINT_TOKENS`

每个模型先做 500 epochs 和 20 个固定 seed 的筛选，但这批结果只用于排除完全失败的实现。通过者再
完成 2000 epochs 和 100 rollouts。validity 和深度反事实统一推迟到 P4，不阻塞当前快速验证。

## P2：扩展架构

只在 P1 无明确赢家或容量受限时加入：

- `ACT7_DEPTH_TRANSFORMER`：判断深度全局 token 建模是否必要；
- `ACT6_LINGBOT_DEPTH`：验证预训练几何先验。

`ACT6` 训练使用 `dp3` 环境；在线 RoboTwin 评测使用已验证的 `RoboTwin-A3 + a3-eval` 组合，不能
直接在当前 robotwin torch 2.4 环境加载现有 xFormers。

## P3：深度处理效果

选出一到两种架构后，再执行 train/deploy 深度输入交叉矩阵：clean GT、sensor-noisy、processed。
点图和 point tokens 必须从当次输入深度重新反投影，不能读取 clean master pointcloud。

主指标是 100-rollout 成功率及置信区间；辅助指标包括分阶段成功率、动作越界率、最大动作跳变、
prior-action L1、深度 zero/shuffle 降幅、ROI 几何误差与三视角一致性。注意力图不用于模型排名。

## GPU 队列原则

- GPU0-3 运行长期共享训练队列；任一架构完成后自动领取剩余架构，全部完成后切换评测队列。
- 新训练只有在 `official_act_rgbd_preflight.json` 为 passed 后才能入队。
- GPU5-7 只在北京时间 2026-09-15 07:00 前参与训练，到时保存并退出；GPU4不使用。
- 所有训练目录包含 `training_last.pt`，重复启动同一实验会自动续训。
- H100 负责正式训练，AutoDL 消费级 GPU 负责只读在线评测；训练产物必须通过 SHA256 清单验收。
- 56 的物理 GPU0 禁令不适用于 H100/AutoDL 单卡实例中的逻辑 `cuda:0`。

## Q0 后继：官方 π0.5 RGB 基线

按 2026-09-15 最新调度，π0.5 不再等待 ACT 评测结束：ACT5 留在 GPU2，ACT 官方评测使用 GPU0-3，
π0.5 立即在 GPU4-7 并行启动。数据严格使用 RoboTwin 官方转换链：原始 HDF5 提取三视角 RGB、
joint、action 和 prompt，再转换为 LeRobot；转换结果必须拒绝任何 depth feature。

- 上游配置：`pi05_aloha_full_base`
- 训练设备：56 服务器物理 GPU4-7，JAX FSDP 四卡
- 预算：官方 20,000 steps，global batch 64，seed 0
- 基础权重：本地 `pi05_base/params`，不访问公网
- 全参判据：`freeze_filter=Nothing`
- 续训：每 1,000 steps 保存，异常退出后从最新 checkpoint 自动恢复
- 实验名：`pi05_stack_blocks_two_rgb_full_seed0`
