# Robot Interaction Labeler V1

第一版可复现流程：

```text
GroundingDINO 检出全部候选
-> 夹爪、关节、RGB-D 和执行相机先验排序
-> VLM 仅处理低分或同类实例歧义
-> SAM2.1 从接触帧向前、向后跟踪
-> 头部与执行腕部人工抽检
```

仓库保留 `depth-process/scripts/` 中经过 19 次抓取复核验证的模型脚本，并在其上提供统一
CLI、ROS bag/LeRobot 数据入口、任务配置和可编辑网页。网页中的人工框优先级高于模型候选；
自动处理完成后再次画框并重跑，即可重新初始化对应 SAM2 轨迹。

## 安装

```bash
cd /ssd/hhw/depth-process
uv venv --python 3.10 .venv-labeler
uv pip install --python .venv-labeler/bin/python -e './interaction-labeler-v1[rosbag,lerobot,models]'
```

SAM2 仍使用服务器上已经配置好的源码和权重。启动前应确保 `sam2` 可以在当前 Python
环境导入，GroundingDINO、Qwen 和 SAM2 权重均为本地路径。

## 一条命令启动

ROS bag 文件或目录：

```bash
.venv-labeler/bin/interaction-labeler run \
  --data /ssd/hhw/depth \
  --format rosbag \
  --workspace /ssd/hhw/annotations/toy_v1 \
  --task interaction-labeler-v1/configs/example_task.yaml \
  --grounding-model /ssd/hhw/depth-process/models/grounding-dino-base \
  --sam2-checkpoint /ssd/hhw/depth-process/models/sam2/checkpoints/sam2.1_hiera_large.pt \
  --vlm-model /ssd/hhw/models/internvla_a1_5/Qwen3.5-2B \
  --port 8769
```

LeRobot 数据集：

```bash
.venv-labeler/bin/interaction-labeler run \
  --data /ssd/hhw/zhuomian/lerobot \
  --format lerobot \
  --workspace /ssd/hhw/annotations/zhuomian_v1 \
  --task interaction-labeler-v1/configs/example_task.yaml \
  --vlm-model /ssd/hhw/models/internvla_a1_5/Qwen3.5-2B \
  --port 8769
```

命令先建立帧与事件索引，再自动打开页面。默认不会立即占用 GPU；可以先在“编辑目标”中填写
目标名称、外观属性、源区、目的区、排除描述和参考图路径，再修正接触帧并在任意相机画目标框，
然后点击“开始自动处理”。使用 `--auto-start` 可在页面启动后立即运行。

LeRobot 若没有明确的抓取事件文件，第一轮使用 episode 中点作为接触帧，并在页面标记警告。
应在运行模型前修正接触帧。原始 ROS bag 会调用 `depth-process` 已验证的提取和机器人时序脚本。

## 分步运行

```bash
interaction-labeler prepare --data DATA --format auto --workspace WORK --task TASK.yaml
interaction-labeler serve --workspace WORK --port 8769
interaction-labeler pipeline --workspace WORK --vlm-model QWEN_PATH
interaction-labeler export --workspace WORK --output annotations.json
```

## 输出

| 文件 | 内容 |
| --- | --- |
| `session.json` | 数据集、动作段、相机角色和开始/接触/释放/结束帧 |
| `annotations.json` | 处理前或处理后的人工框覆盖 |
| `outputs/grounded_candidates/` | GroundingDINO 全部候选 |
| `outputs/interaction_candidates/` | RGB-D 与机器人交互排序、VLM 消歧 |
| `outputs/target_tracks_required/` | 头部和执行腕部逐帧 mask、bbox 与质量标记 |
| `pipeline_status.json`、`logs/` | 可视化页面使用的运行状态和逐阶段日志 |

数据、模型权重和输出目录均不进入 Git。页面服务只读取当前 session 中登记的帧。

## 边界

- V1 对 LeRobot 的自动事件回退是一段 episode 一次抓取；多次抓取应提供更细的时序事件或在页面拆分。
- 人工修改某帧框后需要重新运行 SAM2，已有逐帧 mask 不会仅凭修改 JSON 自动变化。
- 没有完整相机外参时，头部与腕部是基于同步接触、外观和交互证据的软关联，不是三维几何真值。
- VLM 只处理候选歧义，不允许凭任务文本生成不存在的框。
