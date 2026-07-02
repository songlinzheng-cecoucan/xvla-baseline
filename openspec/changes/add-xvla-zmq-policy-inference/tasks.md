# 任务清单

- [x] 从 `add-xvla-smolvla-baseline` 中抽取 ZMQ policy inference 目标。
- [ ] 收集官方 `policy_infer.py` 或 benchmark 通信样例，确认 topic/envelope/payload 字段。
- [ ] 设计并实现 canonical 26D 到 XVLA action payload 的拆分函数。
- [ ] 实现 obs envelope schema 校验。
- [ ] 实现 XVLA obs 到 SmolVLA batch 的 observation adapter。
- [ ] 实现 SmolVLA checkpoint 加载和 single-step inference adapter。
- [ ] 实现 ZMQ SUB/PUB service 主循环。
- [ ] 实现 `test`、`start`、`obs`、`reset` topic handling。
- [ ] 实现 dry-run / replay obs 模式。
- [ ] 增加 action NaN/inf、shape 和 group length 校验。
- [ ] 记录 inference latency 和失败原因。
- [ ] 用本地 mock obs envelope 跑通 service dry-run。
- [ ] 等官方推理仓库发布后，对齐实际字段并更新 spec。
