# Robot Interaction Auto Labeler V2

V2 是独立的任务条件化多视角自动标注项目，复用 V1 已验证的数据适配、SAM2 双向跟踪和
互动页面，并新增头部全场实例盘点、动作阶段识别、交互证据评分、跨视角关联和风险队列。

```text
目标描述输入
-> 头部全场实例盘点
-> 自动识别接触/搬运/释放阶段
-> GroundingDINO 或 SAM3 候选
-> RGB-D、关节、夹爪接触和共同运动评分
-> 单次人工点选纠错
-> SAM2 双向跟踪（SAM3 当前作为开放词汇候选后端）
-> 标定投影或软跨视角关联
-> 只审核低分、遮挡和跟踪漂移帧
```

## 安装

Grounded-SAM2 路线沿用现有 Python 3.10 环境：

```bash
cd /ssd/hhw/depth-process
uv venv --python 3.10 .venv-auto-labeler
uv pip install --python .venv-auto-labeler/bin/python \
  -e './interaction-labeler-v1[rosbag,lerobot,models]' \
  -e './interaction-auto-labeler-v2[grounded-sam2]'
```

SAM3.1 应使用单独的 Python 3.12、PyTorch 2.7+、CUDA 12.6+ 环境，并安装 Meta 官方
`sam3` 仓库。权重需要先获得 Hugging Face 访问权限并下载到本地。V2 不会在运行期间静默
下载权重。

## 启动

```bash
.venv-auto-labeler/bin/auto-labeler-v2 run \
  --data /ssd/hhw/zhuomian/lerobot \
  --format lerobot \
  --workspace /ssd/hhw/annotations/zhuomian_v2 \
  --task interaction-auto-labeler-v2/configs/example_task.yaml \
  --concept-model /ssd/hhw/depth-process/models/grounding-dino-base \
  --sam2-checkpoint /ssd/hhw/depth-process/models/sam2/checkpoints/sam2.1_hiera_large.pt \
  --vlm-model /ssd/hhw/models/internvla_a1_5/Qwen3.5-2B \
  --port 8771
```

页面启动后可以先修正接触帧和目标框，再运行模型。`--auto-start` 会直接启动全流程；
`--skip-inventory` 可跳过较耗时的头部全场抽帧检测。

目标名称、属性、源区、目的区、负例描述、参考图和检测提示词都可在页面的“编辑目标”中修改，
并持久化到当前 workspace。`source_box_xyxy`、`destination_box_xyxy` 可选；只有提供头部 RGB
像素坐标区域后，源区到目的区一项才参与数值评分。LeRobot 若没有夹爪事件或动作阶段元数据，
首次启动沿用 episode 中点并要求页面校正，不会把中点伪装成自动接触检测结果。

## V2 证据

交互实例分数默认采用以下权重，并仅在实际存在的证据上重新归一化：

| 证据 | 权重 | 含义 |
| --- | ---: | --- |
| 接触 | 0.30 | 目标与闭合夹爪的二维距离和 RGB-D 深度一致性 |
| 共同运动 | 0.25 | 闭合后目标与执行末端的相对位置保持 |
| 起止位移 | 0.20 | 目标从动作开始到结束的二维/三维位移 |
| 源区到目标区 | 0.10 | 从任务源区域进入目标区域并在释放后停留 |
| 描述匹配 | 0.10 | 类别、颜色、材质、形状或示例图匹配 |
| 跨视角 | 0.05 | 头部和执行腕部是否为同一物理实例 |

同类玩具的描述分数通常接近。真正区分实例的是接触、闭合后的共同运动和释放后的新位置。
V2 在跟踪前先对头部候选实例评分并写入 `v2_interaction_evidence` 选择，再用该选择初始化
SAM2；跟踪后重新计算可见率、面积跳变和最终风险队列。非执行腕部只作页面对照，不参与实例竞争。

## 跨视角模式

- `soft`：外观、颜色、深度、同步接触和共同运动关联。低 margin 自动进入人工复核。
- `calibrated`：源视角 RGB-D mask 反投影到三维，经头部外参或腕部手眼标定与逐帧 FK
  变换，再投影到目标相机并作为分割提示。该模式要求配置文件提供完整标定。

缺少公共外参时，输出会明确记录 `is_geometric_projection: false`，不会把软匹配伪装成
三维真值。

## 输出

| 路径 | 内容 |
| --- | --- |
| `v2/plan.json` | 可复现阶段图和头部盘点帧计划 |
| `v2/head_inventory.jsonl` | 头部视角全场候选实例 |
| `v2/head_scene_tracks.jsonl` | 稀疏盘点帧间的候选实例关联、起止变化与关联置信度 |
| `v2/instance_evidence.jsonl` | 每个候选的证据分解、分数和 margin |
| `v2/cross_view_links.jsonl` | 跨相机实例 ID、模式和置信度 |
| `v2/review_queue.jsonl` | 低 margin、遮挡、mask 面积跳变等风险项 |
| `outputs/target_tracks_required/` | 最终逐帧 mask 和 bbox |

`contact.py`、`cross_view.py` 和 `scoring.py` 与具体检测器解耦，可以替换成新的关节字段、
标定格式或视觉后端，而不改变导出标注结构。
