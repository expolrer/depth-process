# 后续训练与评测计划

> 本文件保留架构筛选细节。跨机器执行、硬件准入、产物校验和严格状态机以
> `EXECUTION_PLAN_ZH.md` 和 `execution_plan.json` 为唯一准则。

## Q0：立即验证加入深度是否改变表现

只在 `stack_blocks_two/depth_master_clean` 上比较 `ACT0_RGB` 和 `ACT1_EARLY_RGBD`。`ACT1` 只把
metric depth 作为第 4 个图像通道，不输入 validity。两者各训练 500 epochs、batch 8、seed 0，
然后在 AutoDL 4090 D 上使用同一份 30-seed 子集评测。本轮明确推迟 validity、zero/shuffle、
噪声/修复深度、三训练 seed 和其他编码器；先得到“加深度是否值得继续”的方向性结果。

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

通过 P0 后，优先顺序为：

1. `ACT2_DEPTH_CNN`
2. `ACT4_XYZMAP`
3. `ACT5_POINT_TOKENS`

每个模型先做 500 epochs 和 20 个固定 seed 的筛选，但这批结果只用于排除完全失败的实现。通过者再
完成 2000 epochs 和 100 rollouts。validity 和深度反事实统一推迟到 P4，不阻塞当前快速验证。

## P2：扩展架构

只在 P1 无明确赢家或容量受限时加入：

- `ACT3_DEPTH_RESNET`：判断 Depth CNN 是否容量不足；
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

- 当前 GPU4-7 的 FairACT 作业不被新仓库中断。
- 新训练只有在 `official_act_rgbd_preflight.json` 为 passed 后才能入队。
- GPU0 永远禁止 SAPIEN 或训练进程。
- Q0 的 ACT0/ACT1 应在不同 H100 上并行训练，结束后转移到 4090 D 顺序完成 30-episode 评测。
- 新队列先进入 prepared 状态；旧队列自然结束或用户明确切换后再启动。
- H100 负责正式训练，AutoDL 消费级 GPU 负责只读在线评测；训练产物必须通过 SHA256 清单验收。
- 56 的物理 GPU0 禁令不适用于 H100/AutoDL 单卡实例中的逻辑 `cuda:0`。
