# Episode Split 和 Validation Loss 记录

## 目标

训练不应只看 train loss。baseline 阶段至少需要：

- 按 episode 切分 train / val，避免相邻 frame 泄漏。
- 在 held-out validation episodes 上计算与训练一致的 SmolVLA flow-matching loss。

## 已实现工具

```text
scripts/split_lerobot_episodes.py
scripts/eval_smolvla_val_loss.py
```

训练 launcher 也已支持：

```text
--episodes 0,1,2
--split-manifest PATH --split-name train
```

## 当前 3-episode 数据集 split

数据集：

```text
/tmp/robomind_xsens_pick_pipe_lerobot
```

生成命令：

```bash
python scripts/split_lerobot_episodes.py \
  --dataset-root /tmp/robomind_xsens_pick_pipe_lerobot \
  --dataset-repo-id local/robomind_xsens_pick_pipe \
  --output /tmp/robomind_xsens_pick_pipe_lerobot/split_seed1000.json \
  --train-ratio 0.67 \
  --val-ratio 0.33 \
  --seed 1000
```

结果：

```json
{
  "train": [0, 2],
  "val": [1],
  "test": []
}
```

注意：这个 split 只是验证工具链。数据只有 3 个 episode，不足以作为正式模型选择依据。

## Validation Loss Smoke

checkpoint：

```text
/tmp/robomind_smolvla_lora_pick_pipe_500/checkpoints/000500/pretrained_model
```

smoke 命令：

```bash
env \
  -u ALL_PROXY -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY \
  -u http_proxy -u https_proxy \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_CACHE=/tmp/hf_datasets_cache \
  conda run -n lerobot312 python scripts/eval_smolvla_val_loss.py \
    --checkpoint /tmp/robomind_smolvla_lora_pick_pipe_500/checkpoints/000500/pretrained_model \
    --dataset-root /tmp/robomind_xsens_pick_pipe_lerobot \
    --dataset-repo-id local/robomind_xsens_pick_pipe \
    --split-manifest /tmp/robomind_xsens_pick_pipe_lerobot/split_seed1000.json \
    --split-name val \
    --batch-size 1 \
    --max-batches 1 \
    --device cpu \
    --output /tmp/robomind_smolvla_lora_pick_pipe_500/val_loss_smoke.json
```

结果：

```json
{
  "episodes": [1],
  "num_samples": 1,
  "num_batches": 1,
  "mean_loss": 1.5946259498596191,
  "device": "cpu"
}
```

关键修正：

- 不能直接用普通 `LeRobotDataset(...)` 计算 val loss。
- SmolVLA 训练需要 50-step action chunk。
- eval 脚本必须使用 LeRobot 的 `resolve_delta_timestamps`，按 policy config 构造与训练一致的 action delta window。

## 后续正式评估

更多 episode 下载完成后，建议：

```text
train: 80%
val: 20%
test: 暂不启用或保留 10%
```

每轮训练后记录：

```text
train loss
val flow-matching loss
checkpoint path
action output sanity check
```

完整 val loss 应在 GPU 上跑，不建议用 CPU 遍历全部 validation frames。

## Open-Loop Action Inspection

Validation flow-matching loss 能判断模型是否在训练目标上下降，但不够直观。baseline 阶段还需要一个 open-loop action inspection 工具，用 held-out trajectory 的真实 observation 喂给 checkpoint，然后把模型预测 action 和数据集 ground-truth action 对齐比较。

建议新增工具：

```text
scripts/inspect_smolvla_open_loop.py
```

第一版目标：

- 只在 `lerobot312` 环境运行。
- 输入 SmolVLA checkpoint、LeRobotDataset root、repo_id、split manifest 和 split name。
- 加载 checkpoint、policy processor 和 dataset。
- 对 held-out samples 生成模型预测 action。
- 对比预测 action 和 ground-truth action。
- 按 canonical 26D 分组统计误差：

```text
0:7    left_arm
7:14   right_arm
14:20  left_hand
20:26  right_hand
```

推荐命令形态：

```bash
env \
  -u ALL_PROXY -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY \
  -u http_proxy -u https_proxy \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_CACHE=/tmp/hf_datasets_cache \
  conda run -n lerobot312 python scripts/inspect_smolvla_open_loop.py \
    --checkpoint /home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_134ep_5000/checkpoints/005000/pretrained_model \
    --dataset-root /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_lerobot_134 \
    --dataset-repo-id local/robomind_xsens_pick_pipe_134 \
    --split-manifest /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_lerobot_134/split_seed1000.json \
    --split-name test \
    --max-samples 200 \
    --output-dir /home/slzheng/datasets/xvla/reports/open_loop_005000_test
```

第一版输出：

```text
summary.json
per_sample_metrics.csv
per_group_metrics.json
action_error_curve.png
group_mae_bar.png
episode_<id>_action_compare.png
```

`summary.json` 至少记录：

```json
{
  "checkpoint": ".../pretrained_model",
  "dataset_root": "...",
  "dataset_repo_id": "local/robomind_xsens_pick_pipe_134",
  "split_name": "test",
  "episodes": [16, 25],
  "num_samples": 200,
  "mode": "single",
  "full_mae": 0.0,
  "full_mse": 0.0,
  "groups": {
    "left_arm": {"mae": 0.0, "mse": 0.0},
    "right_arm": {"mae": 0.0, "mse": 0.0},
    "left_hand": {"mae": 0.0, "mse": 0.0},
    "right_hand": {"mae": 0.0, "mse": 0.0}
  }
}
```

### Action 对齐模式

SmolVLA 推理侧可以生成 action chunk：

```text
chunk_size = 50
n_action_steps = 50
action_dim = 26
```

inspection 工具应支持两种模式：

```text
--mode single
--mode chunk
```

v1 先实现 `single`：

```text
输入 t 时刻 observation
预测一个 action
与 ground-truth action[t] 对齐
```

v2 再实现 `chunk`：

```text
输入 t 时刻 observation
预测 action[t:t+49]
与 dataset 中同一时间窗口的 future action chunk 对齐
```

`single` 模式更容易解释，适合快速检查 checkpoint 是否输出合理的 26D action。`chunk` 模式更接近 SmolVLA 训练目标，但需要严格处理 episode 边界、padding 和 `resolve_delta_timestamps`。

### 指标解释边界

Open-loop action inspection 只能说明模型在真实轨迹 observation 上复现数据集 action 分布的能力。它不能证明模型具备闭环纠错能力，也不能替代 XVLA benchmark success rate。

当前 reader v1 使用：

```text
state[t] = puppet canonical26[t]
action[t] = puppet canonical26[t]
```

因此 open-loop 指标应被解释为 baseline 阶段的数据/模型 sanity check，而不是最终控制性能。
