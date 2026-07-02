# 设计：XVLA ZMQ Policy Inference Service

## 通信协议

仿真端发布：

```text
SUB tcp://127.0.0.1:5556
topics: test, start, obs, reset
serialization: pickle
```

选手端发布：

```text
PUB tcp://127.0.0.1:5557
topic: action
serialization: pickle
```

envelope 格式：

```python
{
    "schema_version": 1,
    "topic": "obs" 或 "action",
    "episode_id": int,
    "step_id": int,
    "timestamp": float,
    "payload": data,
}
```

动作回复必须沿用当前 obs 的 `episode_id` 和 `step_id`。

## 模块布局

建议新增：

```text
src/xvla_baseline/inference/
  zmq_policy_service.py
  smolvla_policy_adapter.py
  observation_adapter.py
scripts/run_zmq_policy_service.py
```

### `observation_adapter.py`

职责：

- 校验 envelope schema。
- 从 `payload["camera_observations"]["camera_head"]` 读取 RGB。
- 从 `payload["puppet"]` 读取当前机器人状态。
- 构造 SmolVLA 需要的 batch。

第一版如果 XVLA obs 的 puppet state 字段名不稳定，必须 fail fast，并输出可读错误，不得猜字段。

### `smolvla_policy_adapter.py`

职责：

- 加载 checkpoint。
- 加载 processor / normalizer。
- 执行单步推理。
- 输出 canonical 26D action。

第一版优先支持 single-step action。chunk action 可以后续添加，但必须明确缓存策略和 episode reset 行为。

### `zmq_policy_service.py`

职责：

- 建立 ZMQ SUB/PUB socket。
- 响应 `test`、`start`、`obs`、`reset` topic。
- 收到 `reset` 时清空本地 episode state。
- 收到 `obs` 时调用 policy adapter 并发布 action envelope。
- 记录推理耗时和丢弃/失败原因。

## Action Payload

canonical 26D 到 XVLA action payload 的映射固定为：

```python
{
    "left_arm": action[0:7],
    "right_arm": action[7:14],
    "left_hand": action[14:20],
    "right_hand": action[20:26],
}
```

输出必须是普通 Python list 或可 pickle 的数值序列。

## 运行模式

### ZMQ service

```bash
conda run -n lerobot312 python scripts/run_zmq_policy_service.py \
  --checkpoint /home/slzheng/datasets/xvla/runs/.../checkpoints/005000/pretrained_model \
  --sub-url tcp://127.0.0.1:5556 \
  --pub-url tcp://127.0.0.1:5557 \
  --device cuda
```

### Dry-run / replay

```bash
conda run -n lerobot312 python scripts/run_zmq_policy_service.py \
  --checkpoint /home/slzheng/datasets/xvla/runs/.../checkpoints/005000/pretrained_model \
  --replay-obs /path/to/obs_envelope.pkl \
  --device cuda
```

dry-run 必须输出：

- action payload shape。
- inference latency。
- episode_id / step_id 是否沿用。
- 是否出现 NaN/inf。

## 安全边界

第一版可以只做最小动作校验：

- action shape 必须为 26。
- action 不得包含 NaN/inf。
- action group 长度必须分别为 7/7/6/6。

动作限幅、平滑、速度约束和 fallback pose 属于后续控制安全 change。
