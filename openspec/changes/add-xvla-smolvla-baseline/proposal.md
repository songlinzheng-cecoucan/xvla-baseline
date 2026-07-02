# 提案：建立 XVLA SmolVLA Baseline 管线

## 背景

我们需要为 XVLA VLA 操作赛道建立一条可复现的 baseline 路线。比赛动作接口只控制双臂和双灵巧手：

- 左臂：7 DoF
- 右臂：7 DoF
- 左手：6 DoF
- 右手：6 DoF

RoboMIND TienKung Xsens 数据值得作为预适配数据源，因为公开 schema 说明和真实样本都表明它包含：

- `joint_position`：14 维双臂关节
- `end_effector`：12 维双手控制量

这两部分可以稳定拼接成与比赛接口一致的 26 维控制向量。SmolVLA 已经在本地 `lerobot312` 环境中配置完成，适合在 RTX 4090 Laptop 16GB 显存上做 LoRA/PEFT 微调。

## 范围

本变更建立 baseline 阶段的数据与训练契约，覆盖：

- 将 RoboMIND TienKung Xsens 字段映射到 XVLA canonical 26 维状态/动作向量。
- 将选定的 RoboMIND 轨迹转换为 LeRobotDataset 兼容记录。
- 在 `lerobot312` 环境中使用 `lerobot/smolvla_base` 做 LoRA/PEFT smoke training。
- 保持 Isaac Sim/XVLA benchmark 运行环境 `xmimic` 与训练环境分离。
- 产出后续可被 ZMQ `policy_infer.py` 服务加载的 checkpoint 和输入/输出约定。

## 不在范围内

- 完整 XVLA benchmark 策略实现。
- Isaac Sim 任务脚本、遥操作采集或 TianYi2.0 IK 实现。
- 下载完整 RoboMIND 数据集。
- 对所有 TienKung 任务做训练。
- 保证 RoboMIND TienKung/Inspire hand 数据无需 XVLA 同构数据即可迁移到 TianYi2.0/BrainCo2。

## 风险

- RoboMIND TienKung Xsens 手部语义不一定完全等价于 BrainCo2 手部语义。
- RoboMIND `camera_top` 与比赛 `camera_head` 视角不同。
- RoboMIND 任务分布与比赛天平门、托盘、开关、双手传递任务不完全一致。
- 当前代理下载大分片较慢，完整 smoke dataset 下载可能中断。
- SmolVLA LoRA 能拟合源数据不代表能直接通过 XVLA benchmark，最终仍需要 XVLA 同构数据微调或控制保护。

