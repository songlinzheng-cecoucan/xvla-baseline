# 设计：XVLA SmolVLA Baseline 管线

## 环境分工

使用两个独立环境：

- `xmimic`：运行 Isaac Sim 5.1 和 XVLA benchmark 仿真端。
- `lerobot312`：运行 RoboMIND 数据读取、LeRobotDataset 转换、SmolVLA/LoRA 训练和 checkpoint 加载。

这样可以避免 LeRobot、Transformers、PEFT 等训练依赖污染 Isaac Sim 环境。

## Canonical 26 维向量

数据转换、训练和 benchmark action 拆分统一使用以下顺序：

```text
0:7    left_arm
7:14   right_arm
14:20  left_hand
20:26  right_hand
```

RoboMIND TienKung Xsens 到 canonical 26D 的映射：

```text
puppet/joint_position[0:7]   -> left_arm
puppet/joint_position[7:14]  -> right_arm
puppet/end_effector[0:6]     -> left_hand
puppet/end_effector[6:12]    -> right_hand
```

RoboMIND 文档给出的 Xsens 手部 6 维顺序：

```text
[小指, 无名指, 中指, 食指, 拇指0弯曲, 拇指1旋转]
```

转换为 XVLA benchmark action payload 时：

```python
{
    "left_arm": action[0:7],
    "right_arm": action[7:14],
    "left_hand": action[14:20],
    "right_hand": action[20:26],
}
```

## 数据选择

不要下载完整 RoboMIND。先使用小任务子集做 smoke test。

首选 Xsens smoke-test 来源：

```text
benchmark1_1_compressed/h5_tienkung_xsens_1rgb/pick_pipe_place_plate_twice
```

已知问题：即使最小 Xsens 任务也需要多个压缩分片，当前代理下载较慢。可以先使用已经抽出的真实 HDF5 样本和 `robomind_static/` 下的静态 schema 文件实现 reader 骨架。

## LeRobotDataset 特征

第一版只使用一路 RGB 图像和一个 canonical state/action 向量：

```text
observation.images.camera_top
observation.state: shape=(26,)
action: shape=(26,)
task: language_raw
fps: 如果源数据缺少显式 fps，则在 metadata 中记录 inferred/default 状态
```

converter 不得静默伪造缺失字段。如果 task、fps 或图像编码缺失，必须显式 warning 或记录 metadata 标志；关键字段缺失时必须失败。

## Reader 架构

Xsens reader 的详细设计见 `reader-design.md`。

关键决策：

- 不直接复用公开 Gello converter，因为该 converter 只读取 `puppet/joint_position`。
- Xsens 必须组合 `puppet/joint_position` 和 `puppet/end_effector` 才能得到 26D。
- v1 只支持 `source="puppet"`，因为 RoboMIND `robomind.yaml` 对 Xsens 声明 `qpos_arm_key=puppet` 且 `action_arm_key=puppet`。
- v1 使用 absolute target 语义：`state[t] = puppet canonical26[t]`，`action[t] = puppet canonical26[t]`。
- v1 一次读取单条 trajectory 到内存；batch converter 后续按文件顺序处理，不在 reader v1 中实现 streaming。

## 训练

使用本地缓存的 `lerobot/smolvla_base`。第一阶段只做 LoRA/PEFT。

训练 runner 的详细设计见 `smolvla-train-design.md`。

关键决策：

- 不重写 SmolVLA forward、loss、optimizer step 或 PEFT wrapping，训练内核继续复用 LeRobot。
- 仓库内必须提供可测试的 train runner，而不只是拼接 shell command 的 wrapper。
- runner 必须执行 preflight validation，检查 dataset schema、HF cache、CUDA、proxy 和 SmolVLA feature 覆盖项。
- `policy.input_features=null` 和 `policy.output_features=null` 是必要覆盖项，用于让 `smolvla_base` 根据当前 26D dataset 推断输入/输出 feature，而不是沿用 ALOHA 6D/三相机 schema。
- runner 必须写出 `xvla_train_manifest.json`，记录 dataset、训练参数、checkpoint 路径和失败类型。

建议 smoke 设置：

```text
steps: 500-1000
batch_size: 1-2
gradient_accumulation: 按显存调整
peft.method_type: LORA
peft.r: 16
policy.device: cuda
wandb: 默认 disabled/offline
```

smoke training 成功只证明数据与模型路径可运行，不代表 benchmark 策略已经可用。

## 验证

任何长训练前必须完成：

1. 检查至少一个真实 TienKung Xsens HDF5 文件。
2. 确认 `joint_position.shape[-1] == 14`。
3. 确认 `end_effector.shape[-1] == 12`。
4. 确认 `observations/rgb_images/camera_top` 存在。
5. 至少解码一帧 RGB 图像。
6. 确认 `language_raw` 存在并可解码为非空任务文本。
7. 保存一个小型转换数据集，并用 LeRobot dataloader 打开。
8. 运行一次短 SmolVLA LoRA smoke training。
