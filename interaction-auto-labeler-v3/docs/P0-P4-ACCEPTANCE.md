# V3 P0-P4 验收清单

| 阶段 | 可执行验收 | 不应误读 |
| --- | --- | --- |
| P0 | 事件图 round-trip；LeRobot v3 layout 校验；稳定分片；队列宕机恢复；金标确定性抽样 | 元数据 hash 不是文件内容 hash，发布版本应使用 `--hash-mode full` |
| P1 | 三级边界嵌套；语言实体引用合法；边界误差/F1 在金标集统计 | 无多模态 NPZ 时只生成 V2 事件的保守层级 |
| P2 | 金标 isotonic 校准单调；选择性 precision/coverage；GVL 时序；异常门控 | 未校准 raw score 不等于真实正确率；缺 GVL 时不能自动接收 |
| P3 | FK 合成测试；跨视角 IoU/像素误差；点云融合；6D 轨迹跳变 | 无外参/FK 时的软匹配不是三维重投影；PCA 朝向不是规范 6D pose |
| P4 | 纠错到 COCO/JSONL；A/B 同 split/seed/checkpoint/budget；配对 bootstrap | 离线未来轨迹只能作监督，不能作为在线策略必需输入 |

发布门槛建议：金标集至少双人复核，自动接收阈值以目标 precision 为约束，数据和标注各自版本化；
每次模型/阈值变更都冻结一份 untouched gold set，避免在同一金标上反复调参后高估性能。
