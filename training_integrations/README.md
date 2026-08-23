# 将目标框和 mask 用于 ACT、pi0.5 与 VLA 训练

自动标注结果应先导出为 LeRobot sidecar，不直接修改原始 RGB 视频：

```bash
python -m training_integrations.export_target_sidecar \
  --session /ssd/hhw/annotations/zhuomian_v2/session.json \
  --tracks /ssd/hhw/annotations/zhuomian_v2/outputs/target_tracks_required/track_index.jsonl \
  --output /ssd/hhw/zhuomian/lerobot/meta/target_annotations.jsonl
```

每帧包含 `bbox_xyxy`、归一化框、可见性、mask 路径、实例 ID、动作阶段、置信度和复核标志。
训练 DataLoader 使用 `(episode_index, frame_index, camera)` 与原 LeRobot 行连接。

可直接复用的代码入口：

- `act_adapter.py`：生成 ACT 使用的归一化 ROI 坐标、视觉 patch 监督和可选 ROI token。
- `openpi_adapter.py`：为 OpenPI 数据变换增加全图保留的目标 crop、ROI dropout 和有效位。
- `target_annotations.py`：sidecar 索引、框归一化、patch target、ROI crop 和辅助定位损失。

## 推荐顺序

### 1. 辅助目标定位损失，首选

将框或 mask 映射到视觉 patch 网格，对 ACT/VLA 的视觉 token 增加一个轻量定位 head：

```python
target = box_to_patch_target(annotation.bbox_normalized, grid_h, grid_w)
loss = action_loss + lambda_roi * roi_auxiliary_loss(roi_logits, target, visible)
```

部署时丢弃定位 head，不需要输入检测框。建议 `lambda_roi=0.02~0.10` 起步，并分别报告动作
损失、定位 IoU 和真机指标。这种方式最不容易产生训练与部署分布差异。

### 2. 目标裁剪作为额外视觉流

保留完整头部/腕部图像，同时从框扩张 10% 到 25% 后生成目标 crop：

- ACT：将 crop 经同一个视觉骨干编码，作为额外视觉 token 与全图 token、joint token 一同输入 encoder。
- pi0.5/OpenPI：在数据变换阶段增加 `observation.images.<camera>_target_crop`，再按仓库现有多相机
  图像 token 方式输入 PaliGemma/VLM 主干。
- 其他 VLA：把 crop 当作额外 camera/view，而不是用黄色框污染原图。

训练时应以 30% 到 50% 概率删除 crop/框，并保留全图，防止策略依赖一个部署时可能失败的检测器。

### 3. 框坐标或 ROI token 作为显式输入

将每个视角的 `[cx, cy, w, h, visible]` 经过 MLP 变成 ROI token：

- ACT：ROI token 与 joint、camera token 一起送入 Transformer encoder。
- pi0.5：ROI token 插入视觉 token 与语言 token 之后、action expert 之前，或作为独立 observation token。
- 通用 VLA：可进一步加入目标深度中位数和三维质心，但单位、坐标系和缺失值 mask 必须固定。

该方案只有在部署端也运行 GroundingDINO/SAM3/任务检测器时才成立。否则训练时有框、推理时没框，
模型性能通常会下降。

### 4. ROI 引导的数据增强与采样

- 随机遮挡尽量不完全覆盖目标 mask；另保留一组故意遮挡目标的鲁棒性样本。
- 可提高接触、搬运和释放阶段采样权重，但验证集保持真实时序分布。
- 使用 ROI 内深度有效率、边缘完整率和遮挡恢复质量筛除异常深度，不按注意力热力图直接删帧。

## ACT 接入点

ACT 默认输入 RGB 和 joint/state，最稳妥的改动是视觉编码器后增加 ROI 定位 head。需要更强目标条件化时，
再加入全图 + crop 双视觉流。不要先把未来轨迹推断出的框拼到 joint state；这会让行为克隆获得在线不可用信息。

## pi0.5/OpenPI 接入点

pi0.5 已有 prompt 条件化。框主要解决同类实例定位，不应替代 prompt：

```text
完整 RGB token + 目标 crop token + prompt token + proprio token -> action expert
```

第一轮实验建议保持预训练模型结构不变，只增加目标 crop observation 和辅助定位损失；随后再做 ROI token
架构消融。训练、验证和部署必须使用同一 camera key、归一化方式和 ROI 缺失策略。

## 防止时序泄漏

离线标注会利用接触后的共同运动和动作结束位置确定目标，这些是“未来证据”。它们可以生成训练监督，
但不能在当前帧作为必需输入，除非在线系统能仅凭当前及历史帧产生同样的框。推荐做法：

- 未来证据只用于确定 ground-truth instance、辅助损失和评估 ROI。
- 显式框输入必须来自因果在线检测结果，并记录 `source=online_detector`。
- 按 episode 划分训练/验证集，不能让同一抓取的相邻帧跨集合。
- 对遮挡或不可见帧使用 `visible=false`，不要沿用最后一个框伪造标签。

## 建议消融

| 版本 | 输入 | 额外监督 | 推理需要检测器 |
| --- | --- | --- | --- |
| A | 原始 RGB-D + joint + prompt | 无 | 否 |
| B | 与 A 相同 | ROI/mask 辅助定位 | 否 |
| C | 全图 + 目标 crop | ROI 辅助定位 | 是，或使用高比例 ROI dropout |
| D | 全图 + crop + ROI token | ROI 辅助定位 | 是 |

先比较 A/B，可验证标注是否改善视觉表征；B 有稳定收益后再比较 C/D。
