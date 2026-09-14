# 基于官方 ACT 的 RGB-D 架构设计

## 1. 核心边界

官方 ACT 的动作预测路径定义为：

```text
action history -> CVAE latent encoder -> z
RGB backbone features + z + joint state
  -> official Transformer encoder/decoder
  -> 50 action queries -> 50 x 14 joint actions
```

本项目只允许在 `RGB backbone features` 的生成位置加入深度。以下部分全部冻结为共同实验条件：

- action CVAE 的输入、latent 维度、重参数化和 KL 公式；
- proprioception projection；
- Transformer encoder/decoder；
- action query、action head、padding loss；
- 动作归一化、反归一化和 50 步 chunk 的执行语义。

这种边界比“重新写一个类似 ACT 的 Transformer”更重要。旧 FairACT 与官方实现同时存在
分辨率、位置编码、优化器和动作执行差异，无法把成功率变化单独归因于深度。

## 2. 深度前端

### ACT1：四通道早期拼接

将 ImageNet 归一化 RGB 与 `clip(depth_m, 0, 2) / 2` 拼接。ResNet18 第一层由 3 通道扩为
4 通道，新增深度卷积核初始化为 0，因此初始化时输出与 RGB 骨干一致，之后由训练学习深度增量。

### ACT2：共享双流 ResNet18

三个视角共享一个 RGB ResNet18，并共享一个单通道 Depth ResNet18。以 RGB token 为 Query、Depth
token 为 Key/Value 做 cross-attention，再通过门控残差写回 RGB token。本轮只输入 clean metric depth。

### ACT3：逐视角六分支 ResNet18

头部、左腕、右腕分别使用独立 RGB ResNet18 和独立单通道 Depth ResNet18，共六个视觉分支。
显式相机 one-hot 只用于把官方相机循环路由到正确分支，不作为几何输入参与特征学习。

### ACT4：XYZ 点图

使用相机内参把 metric depth 反投影为 `(x, y, z)` 并输入几何 CNN。XYZ 点图
保留像素邻接关系，更适合堆叠、插入和接触边界等精细几何任务。

### ACT5：Point Tokens

从同一张深度图反投影 XYZ 后按规则网格采样，使用共享 point MLP 形成无序点 token，再与 RGB
token 交叉注意力。噪声/修复实验必须由对应版本的深度重新生成点，禁止读取 clean GT pointcloud，
否则会产生几何信息泄漏。官方 DP3 只作为外部点云方法参考，不与 ACT5 混为同一基线。

### ACT6：LingBot-Depth Token

冻结官方 LingBot-Depth v0.5，仅训练 1024->512 adapter 和融合层。第一轮不解冻 LingBot；若
冻结版通过反事实依赖门槛，再进行最后若干层解冻对照。

### ACT7：Depth Transformer

用 stride-32 patch embedding 将 clean metric depth 转为与 RGB 相同的空间 token，经小型
Transformer encoder 后融合。它是“Transformer 处理深度”的对照，不改用 action DiT；
action DiT 会同时改变动作生成器，不能用于第一轮单变量比较。

## 3. 数据与评测设计

架构筛选只使用 clean RGB + clean GT Depth。同一模型不能在训练时看到 processed depth 而在
部署时换 clean depth，反之亦然。深度处理效果在选定架构后用 3 x 3 矩阵评测：

```text
train depth:  clean | noisy | processed
deploy depth: clean | noisy | processed
```

必须额外加入三项反事实：

- `zero_depth`：深度和值有效性全部置零；
- `shuffle_depth`：在 batch/episode 间打乱深度，保持边际分布但破坏语义对齐；
- `shuffle_views`：头部与腕部深度互换，检查跨视角对齐依赖。

若真实深度、零深度和打乱深度成功率近似，即使门控值或注意力热图看起来集中，也不能声称模型
使用了深度。最终证据以任务成功率、几何扰动鲁棒性和反事实差值为主，注意力仅作为解释材料。

## 4. 推荐迭代顺序

当前先让八种架构在同一份 `stack_blocks_two/depth_master_clean` 上完成相同预算训练。该轮只筛选
架构，不引入 validity、噪声或深度修复变量；胜出架构再进入多任务与 clean/noisy/processed 实验。
