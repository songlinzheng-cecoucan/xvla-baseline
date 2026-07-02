## ADDED Requirements

### Requirement: ZMQ Envelope Handling

policy inference service MUST read XVLA benchmark envelopes and reply with action envelopes that preserve the latest observation identity.

#### Scenario: 回复最新 obs 的 episode 和 step

- **GIVEN** service 收到一个 `topic="obs"` 的 envelope
- **AND** envelope 包含 `schema_version=1`、`episode_id` 和 `step_id`
- **WHEN** service 发布 `topic="action"` 的 envelope
- **THEN** action envelope 必须沿用 obs 的 `episode_id`
- **AND** 必须沿用 obs 的 `step_id`
- **AND** 必须设置新的 `timestamp`

#### Scenario: 拒绝非法 envelope

- **GIVEN** service 收到缺少 `schema_version`、`topic`、`episode_id`、`step_id` 或 `payload` 的消息
- **WHEN** service 校验 envelope
- **THEN** 必须记录结构化错误
- **AND** 不得发布 action

### Requirement: Canonical 26D Action Payload

policy inference service MUST convert canonical 26D model output into XVLA benchmark action payload.

#### Scenario: 拆分 26D action

- **GIVEN** 模型输出一个长度为 26 的 action
- **WHEN** service 构造 action payload
- **THEN** `action[0:7]` 必须写入 `left_arm`
- **AND** `action[7:14]` 必须写入 `right_arm`
- **AND** `action[14:20]` 必须写入 `left_hand`
- **AND** `action[20:26]` 必须写入 `right_hand`
- **AND** 每个字段必须是可 pickle 的数值 list

#### Scenario: 拒绝非法 action

- **GIVEN** 模型输出 action 维度不是 26
- **OR** action 包含 NaN 或 inf
- **WHEN** service 构造 action payload
- **THEN** 必须标记推理失败
- **AND** 不得发布非法 action

### Requirement: SmolVLA Checkpoint Inference Adapter

policy inference service MUST provide a checkpoint adapter that turns one XVLA observation into one canonical 26D action.

#### Scenario: 单步 checkpoint 推理

- **GIVEN** 一个本地 SmolVLA LoRA checkpoint
- **AND** 一个合法 XVLA obs payload
- **WHEN** inference adapter 运行 single-step 推理
- **THEN** 必须加载 policy、processor 和 normalizer
- **AND** 必须从 obs 中读取 RGB 图像、当前 26D state 和 task 文本或默认 task
- **AND** 必须返回长度为 26 的 action

### Requirement: Topic Lifecycle

policy inference service MUST handle XVLA benchmark lifecycle topics explicitly.

#### Scenario: episode reset

- **GIVEN** service 收到 `topic="reset"`
- **WHEN** reset 被处理
- **THEN** service 必须清空本地 episode state
- **AND** 不得继续使用 reset 前缓存的 action chunk

#### Scenario: communication warmup

- **GIVEN** service 收到 `topic="test"`
- **WHEN** warmup 消息被处理
- **THEN** service 必须记录通信可用
- **AND** 不得把 test payload 当作 obs 推理

### Requirement: Dry-Run Replay

policy inference service MUST support replaying a saved obs envelope without connecting to Isaac Sim.

#### Scenario: replay one obs envelope

- **GIVEN** 一个 pickle 保存的 obs envelope
- **AND** 一个本地 checkpoint
- **WHEN** service 以 replay 模式运行
- **THEN** 必须输出 action payload group lengths
- **AND** 必须输出 inference latency
- **AND** 必须报告 action 是否包含 NaN/inf
