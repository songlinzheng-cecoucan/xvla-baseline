# xvla-smolvla-baseline Specification

## Purpose
记录 XVLA baseline 的稳定数据、训练和离线评估契约：RoboMIND TienKung Xsens 到 canonical 26D 的映射、LeRobotDataset 兼容性、episode-level split、SmolVLA LoRA smoke training、validation loss 和 single-step open-loop action inspection。

多任务 Xsens 数据选择由 `add-xvla-xsens-multitask-baseline` 管理；ZMQ policy inference service 由 `add-xvla-zmq-policy-inference` 管理。

## Requirements
### Requirement: Canonical XVLA 26D 控制向量

baseline 管线 MUST 使用统一的 26 维向量表示机器人 state 和 action，顺序固定为：

```text
left_arm[7], right_arm[7], left_hand[6], right_hand[6]
```

#### Scenario: 将 canonical vector 转为 benchmark action

- **GIVEN** 一个长度为 26 的模型输出向量
- **WHEN** 该向量被转换为 XVLA benchmark action payload
- **THEN** 索引 `0:7` 必须填入 `left_arm`
- **AND** 索引 `7:14` 必须填入 `right_arm`
- **AND** 索引 `14:20` 必须填入 `left_hand`
- **AND** 索引 `20:26` 必须填入 `right_hand`

### Requirement: RoboMIND TienKung Xsens 字段映射

converter MUST 根据 RoboMIND 文档和真实样本验证过的语义，将 TienKung Xsens 字段映射到 canonical 26D 向量。

#### Scenario: 转换 Xsens puppet state

- **GIVEN** 一个 HDF5 trajectory，其中 `puppet/joint_position` 的 shape 为 `(T, 14)`
- **AND** `puppet/end_effector` 的 shape 为 `(T, 12)`
- **WHEN** converter 创建 `observation.state`
- **THEN** `joint_position[:, 0:7]` 必须映射到 `left_arm`
- **AND** `joint_position[:, 7:14]` 必须映射到 `right_arm`
- **AND** `end_effector[:, 0:6]` 必须映射到 `left_hand`
- **AND** `end_effector[:, 6:12]` 必须映射到 `right_hand`

#### Scenario: 拒绝非预期 Xsens 维度

- **GIVEN** 一个 HDF5 trajectory 的 Xsens `joint_position` 最后一维不是 14
- **OR** Xsens `end_effector` 最后一维不是 12
- **WHEN** 请求转换
- **THEN** converter 必须抛出显式 schema error
- **AND** 不得静默截断、补零或重排维度

### Requirement: Xsens Reader v1 语义

reader v1 MUST 只支持 TienKung Xsens puppet 数据源，并使用 absolute target 语义。

#### Scenario: 读取 v1 state/action

- **GIVEN** 一个合法的 TienKung Xsens HDF5 trajectory
- **WHEN** reader v1 读取 trajectory
- **THEN** `state[t]` 必须等于 `puppet` canonical 26D 在时间步 `t` 的值
- **AND** `action[t]` 必须等于 `puppet` canonical 26D 在时间步 `t` 的值
- **AND** reader v1 不得暴露或默认使用 `master` 作为 action 来源

### Requirement: Xsens Schema Validation

reader MUST 在返回数据前验证关键字段、shape、dtype、task 文本和图像可解码性。

#### Scenario: 合法 Xsens trajectory 通过 validation

- **GIVEN** 一个 HDF5 trajectory 包含非空 `language_raw`
- **AND** 包含数值型 `puppet/joint_position`，shape 为 `(T, 14)` 且 `T > 0`
- **AND** 包含数值型 `puppet/end_effector`，shape 为 `(T, 12)`
- **AND** 包含 `observations/rgb_images/camera_top`，长度为 `T`
- **AND** canonical 26D 不包含 NaN 或 inf
- **AND** 第 0 帧 RGB 图像可以解码
- **WHEN** reader validation 执行
- **THEN** validation 必须通过
- **AND** reader 必须记录 task、num_frames、image size、source、action_mode 和路径 metadata

#### Scenario: 非法 Xsens trajectory 返回结构化错误

- **GIVEN** 一个 HDF5 trajectory 缺失必要字段、维度不匹配、dtype 非数值、task 缺失、或第 0 帧 RGB 无法解码
- **WHEN** reader validation 执行
- **THEN** reader 必须抛出 `XsensSchemaError`
- **AND** 错误信息必须包含 `path`、`field`、`expected`、`actual` 和 `reason`

### Requirement: 单 RGB 相机输入

第一版 baseline dataset MUST 使用 RoboMIND Xsens 的 `observations/rgb_images/camera_top` 作为图像观测。

#### Scenario: camera_top 图像流存在

- **GIVEN** 一个 TienKung Xsens HDF5 trajectory
- **WHEN** converter 读取图像观测
- **THEN** 必须要求存在 `observations/rgb_images/camera_top`
- **AND** 必须能将 encoded RGB frame 解码或引用为 LeRobot 兼容格式
- **AND** 不得在 reader 层自动把 `camera_top` 重命名为 benchmark 的 `camera_head`

### Requirement: Task 和 Metadata

reader MUST 从 HDF5 中读取明确的任务文本和 metadata，不得静默猜测任务语义。

#### Scenario: 读取 task 文本

- **GIVEN** 一个 HDF5 trajectory 包含 `language_raw[0]`
- **WHEN** reader 创建 `XsensTrajectory`
- **THEN** `task` 必须来自 `language_raw[0]`
- **AND** 文件夹名只能作为 metadata，不得作为默认 task fallback

#### Scenario: 缺失 task 文本

- **GIVEN** 一个 HDF5 trajectory 缺失 `language_raw`
- **OR** `language_raw[0]` 为空或无法解码
- **WHEN** reader 创建 `XsensTrajectory`
- **THEN** reader 必须抛出 schema error

### Requirement: LeRobotDataset 兼容性

转换后的 smoke dataset MUST 能被 LeRobot training code 加载，不需要修改模型代码。

#### Scenario: 加载转换后的 dataset

- **GIVEN** 本地磁盘上存在一个转换后的 dataset
- **WHEN** LeRobot dataloader 打开该 dataset
- **THEN** 每个 sample 必须暴露一路图像观测、26D state、26D action，以及 task 文本或显式 task metadata

### Requirement: Episode-Level Split

baseline evaluation MUST 使用 episode-level split，避免相邻 frame 泄漏到 validation 或 test。

#### Scenario: 生成 train/val split

- **GIVEN** 一个转换后的 LeRobotDataset 包含多个 episode
- **WHEN** split 工具生成训练和验证划分
- **THEN** 同一个 episode 的所有 frame 必须只属于一个 split
- **AND** split manifest 必须记录 dataset root、repo_id、seed、episode 总数和各 split episode id
- **AND** 不得按 frame 随机切分

### Requirement: Validation Flow-Matching Loss

baseline training path MUST 支持在 held-out validation episodes 上计算 SmolVLA flow-matching loss。

#### Scenario: 计算 validation loss

- **GIVEN** 一个 SmolVLA checkpoint
- **AND** 一个转换后的 LeRobotDataset
- **AND** 一个 validation episode 列表
- **WHEN** validation loss 脚本运行
- **THEN** 必须加载 checkpoint、policy processor 和 dataset
- **AND** 必须调用 policy forward 计算与训练一致的 flow-matching loss
- **AND** 必须输出 mean loss、样本数、episode 列表和 checkpoint 路径

### Requirement: Open-Loop Action Inspection

baseline evaluation MUST 支持在 held-out episodes 上比较 SmolVLA 预测 action 和 dataset ground-truth action。

#### Scenario: 单步 open-loop action 对比

- **GIVEN** 一个 SmolVLA checkpoint
- **AND** 一个转换后的 LeRobotDataset
- **AND** 一个 held-out episode 列表
- **WHEN** open-loop inspection 脚本以 `--mode single` 运行
- **THEN** 脚本必须加载 checkpoint、policy processor 和 dataset
- **AND** 必须用真实 observation 生成预测 action
- **AND** 必须将预测 action 与同一 sample 的 ground-truth `action` 对齐
- **AND** 必须计算 full 26D MAE/MSE
- **AND** 必须分别计算 `left_arm`、`right_arm`、`left_hand` 和 `right_hand` 的 MAE/MSE
- **AND** 必须输出 `summary.json`、逐 sample 指标和至少一张误差图

#### Scenario: action 分组误差

- **GIVEN** 一个预测 action 和一个 ground-truth action，二者 shape 均为 `(26,)`
- **WHEN** open-loop inspection 计算分组指标
- **THEN** 索引 `0:7` 必须统计为 `left_arm`
- **AND** 索引 `7:14` 必须统计为 `right_arm`
- **AND** 索引 `14:20` 必须统计为 `left_hand`
- **AND** 索引 `20:26` 必须统计为 `right_hand`
- **AND** 不得把不同 group 的误差混合后替代分组指标

#### Scenario: open-loop 指标解释边界

- **GIVEN** open-loop inspection 生成了 action MAE/MSE 指标
- **WHEN** 记录或展示结果
- **THEN** 结果必须标注其为 open-loop imitation sanity check
- **AND** 不得把 open-loop action error 解释为 XVLA benchmark success rate
- **AND** 不得声称该指标能证明闭环控制成功

### Requirement: SmolVLA LoRA Smoke Training

baseline training path MUST 支持在长训练前先运行短 SmolVLA LoRA smoke test。

#### Scenario: 运行训练 smoke test

- **GIVEN** 一个 tiny converted LeRobotDataset
- **AND** 本地可用 `lerobot/smolvla_base` 权重
- **WHEN** 在 `lerobot312` 中执行 LoRA smoke training
- **THEN** 训练必须能在 CUDA 上启动
- **AND** 必须完成至少一个 checkpoint 或显式 dry-run validation
- **AND** 失败时必须能区分问题属于 data schema、显存、依赖还是 model loading

### Requirement: Runtime 环境分离

Isaac Sim runtime 依赖 MUST 与训练依赖保持分离。

#### Scenario: 分别运行仿真和训练工具

- **GIVEN** 本地存在 `xmimic` 和 `lerobot312` 两个 conda 环境
- **WHEN** 运行 benchmark 仿真端
- **THEN** 必须使用 `xmimic`
- **AND** 当运行 dataset conversion 或 SmolVLA training
- **THEN** 必须使用 `lerobot312`
