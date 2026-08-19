# HHW RGB-D 深度处理项目

本项目用于从 `/ssd/hhw/depth` 下的 5 个 ROS1 bag 中提取、处理并对比物理尺度深度数据。
项目不会修改源 rosbag，也不会修改服务器上的官方模型仓库。

## 相机处理方式

- `cam_h`：Orbbec Gemini-335L。录制的深度图已经位于彩色相机光学坐标系中。
- `cam_l`、`cam_r`：Intel RealSense D405。使用 rosbag 中记录的深度/彩色相机内参、畸变参数和 `/tf_static`，将原始校正深度投影到 RGB 图像坐标系。
- RGB 与深度帧按照最接近的 bag 接收时间戳配对。绝大多数数据流的时间偏差 P95 小于 16 ms，低于约 33 ms 帧周期的一半。
- 深度图以无损 `uint16` PNG 保存，单位为毫米；像素值 `0` 表示无效深度。

## 已实现的处理结果

| 目录 | 用途 |
| --- | --- |
| `depth_raw_mm` | rosbag 中压缩深度 PNG 的原始无损数据 |
| `depth_aligned_rgb_mm` | 已配准到 RGB 坐标系的 D405 深度图 |
| `rgb_guided` | 中值离群点抑制、有限范围孔洞修复和 RGB 引导滤波 |
| `temporal_rgb_guided` | 在 RGB 引导结果上加入光流对齐的时序稳定处理 |
| `lingbot_v05` | LingBot-Depth v0.5 官方模型输出的纯模型物理尺度深度 |
| `depth_anything_v2_fused` | 将 Depth-Anything-V2 相对深度逐帧标定后，与传感器深度融合 |
| `lingbot_v05_sensor_fused` | 保留传感器原始有效像素，仅使用 LingBot 结果填补空洞 |
| `ai_consensus_fused` | 保留传感器深度，仅在 LingBot 与 Depth-Anything 预测一致的位置填补空洞 |
| `lingbot_cross_attention` | LingBot 深度查询到 RGB Token 的跨模态注意力热力图 |
| `lingbot_depth_token_attention` | LingBot CLS 查询到深度 Token 的注意力热力图 |

相比完全使用模型预测替换传感器深度，保留传感器有效像素的融合结果和双模型一致性结果更适合作为 VLA 训练候选数据。纯模型输出仍会保留，供分析和对比使用。

## 主要命令

```bash
cd /ssd/hhw/depth-processing

.venv/bin/python scripts/inspect_bags.py /ssd/hhw/depth \
  --output reports/bag_inventory.json

.venv/bin/python scripts/extract_rgbd.py /ssd/hhw/depth \
  --output-root outputs/extracted --workers 16

.venv/bin/python scripts/align_depth_to_rgb.py \
  --input-root outputs/extracted

.venv/bin/python scripts/process_classical.py \
  --input-root outputs/extracted --output-root outputs/processed

OVERWRITE=1 BATCH_SIZE=4 scripts/run_lingbot_8gpu.sh
OVERWRITE=1 BATCH_SIZE=8 scripts/run_depth_anything_8gpu.sh
GPU_IDS=0,6,7 BATCH_SIZE=4 scripts/run_lingbot_attention_gpus.sh

.venv/bin/python scripts/fuse_ai_outputs.py \
  --input-root outputs/extracted --processed-root outputs/processed

.venv/bin/python scripts/generate_comparisons.py \
  --input-root outputs/extracted \
  --processed-root outputs/processed \
  --output-root outputs/comparisons

.venv/bin/python scripts/compute_depth_norm_stats.py \
  --input-root outputs/extracted \
  --processed-root outputs/processed \
  --output outputs/reports/norm_stats.json

.venv/bin/python scripts/validate_outputs.py \
  --project-root /ssd/hhw/depth-processing \
  --lingbot-checkpoint /ssd/hhw/lingbot-depth/checkpoint/lingbot-vla-v2-depth/model.pt \
  --depth-anything-checkpoint /ssd/hhw/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
  --output outputs/reports/validation_report.json

.venv/bin/python scripts/build_video_viewer.py \
  --project-root /ssd/hhw/depth-processing \
  --workers 8

python3 viewer/serve_viewer.py --host 127.0.0.1 --port 8765
```

The video viewer presents synchronized head, left-wrist, and right-wrist RGB streams with two
independently selectable depth-processing rows. Videos use a fixed 0.2-4.0 m Turbo color scale;
invalid depth is shown near black. The bundled server supports byte-range requests for responsive
seeking.

## 注意力热力图

`process_lingbot_attention.py` 从官方 LingBot-Depth v0.5 RGB-D ViT 中提取真实模型注意力。
脚本通过编码器第 20–23 层的 `qkv` 投影精确重算所需的多头注意力矩阵行：

- 跨模态注意力：有效深度查询到全部 RGB Key 的注意力，聚合层、注意力头和采样查询。
- 深度 Token 注意力：CLS 查询到全部深度 Key 的注意力，聚合层和注意力头。
- 原始相对分数以无损 16-bit PNG 保存，页面叠加图以 Inferno 色标保存为 JPEG。
- 可视化按每帧 P5–P99.5 归一化；归一化前的概率统计保存在分片报告中。

这些热力图解释的是 LingBot-Depth RGB-D 编码器，不是 VLA 动作策略中受任务提示词调节的注意力，
因此不能直接解释为抓取或操作决策区域。

## 模型位置

- LingBot-Depth 仓库：`/ssd/hhw/lingbot-depth`
- LingBot-Depth v0.5 权重：
  `/ssd/hhw/lingbot-depth/checkpoint/lingbot-vla-v2-depth/model.pt`
- Depth-Anything-V2 仓库：`/ssd/hhw/Depth-Anything-V2`
- Depth-Anything-V2-Small 权重：
  `/ssd/hhw/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth`

## 测试结果说明

运行以下命令可执行项目的自动化测试：

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests
```

测试输出中的 `2 passed` 表示 pytest 一共执行了 2 个自动化测试，并且两个测试都通过了，没有出现失败或错误：

1. `tests/test_rosbag_extract.py`：验证 RGB 与深度帧的最近时间戳配对逻辑。
2. `tests/test_fusion.py`：验证 Depth-Anything 相对逆深度的物理尺度标定，以及传感器深度与模型预测的融合逻辑。

`2 passed` 不表示只处理了 2 帧，也不代表仅凭两个单元测试就证明全部深度图的视觉质量完全正确。完整数据集是否有漏帧、输出尺寸是否一致、模型权重是否匹配等内容，由 `scripts/validate_outputs.py` 和生成的 `outputs/reports/validation_report.json` 另外检查。

## 方法适用范围

ClearGrasp 和 TransCG 需要透明物体监督数据或合适的预训练权重，不能直接作为适用于所有图像帧的通用滤波器。MonoGS/NeRF 需要相机轨迹，并通常假设场景基本静态。在缺少掩码、位姿和对应场景假设时，不应把这些方法标记为已经在本批 rosbag 上完成。

当前已经导出的 RGB、原始/配准深度、相机内参、坐标变换、时间戳和清单文件，可以支持后续接入这些处理后端。
