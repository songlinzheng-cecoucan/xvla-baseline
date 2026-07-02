# 提案：实现 XVLA ZMQ Policy Inference Service

## 背景

`add-xvla-smolvla-baseline` 已经完成离线数据、训练和评估链路，但还没有把 checkpoint 接入 XVLA benchmark 的 ZMQ 通信协议。

XVLA 仿真端会通过 ZMQ 发布 `start`、`obs`、`reset` 等 envelope，选手侧需要在收到最新 `obs` 后返回 `action` envelope。动作 payload 必须拆成：

```text
left_arm[7]
right_arm[7]
left_hand[6]
right_hand[6]
```

本 change 只定义和实现推理服务边界，不重新设计训练或数据转换。

## 范围

本 change 覆盖：

- 加载本地 SmolVLA/LoRA checkpoint 和 processor。
- 将 XVLA `obs` envelope 转为模型输入。
- 将模型输出的 canonical 26D action 拆成 benchmark action payload。
- 通过 ZMQ 订阅仿真端 `obs`，并发布 `action`。
- 沿用最新观测的 `episode_id` 和 `step_id`，避免动作被平台判定为过期。
- 提供 dry-run / replay 模式，用本地保存的 obs 样本验证推理 IO。

## 不在范围内

- 训练新模型或下载新数据。
- Isaac Sim 任务成功率优化。
- TianYi2.0/BrainCo2 与 TienKung/Inspire hand 的精确动作重映射。
- IK、碰撞检测、安全夹爪控制或动作后处理策略。
- 修改 XVLA benchmark 官方通信协议。

## 风险

- RoboMIND `camera_top` 和 XVLA `camera_head` 视角不同，直接推理可能分布外。
- SmolVLA 离线 action 误差不代表闭环成功率。
- 输出动作如果没有限幅、平滑或频率控制，可能导致仿真异常或超时失败。
- 官方推理仓库尚未完整发布时，协议细节可能变化；实现必须把 schema/version/topic 校验写清楚。
