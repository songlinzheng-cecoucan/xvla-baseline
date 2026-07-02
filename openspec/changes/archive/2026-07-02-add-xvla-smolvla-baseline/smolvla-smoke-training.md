# SmolVLA LoRA Smoke Training 记录

## 目标

验证以下链路可以在本地 `lerobot312` 环境中启动并完成最小训练：

```text
RoboMIND Xsens HDF5
-> canonical 26D LeRobotDataset
-> lerobot/smolvla_base
-> LoRA fine-tuning
-> checkpoint
```

该 smoke test 只验证数据、模型、PEFT 和 CUDA 运行路径连通，不代表策略已经具备 XVLA benchmark 成功率。

## 输入数据

转换后的 tiny dataset：

```text
/tmp/robomind_xsens_lerobot_smoke
```

数据统计：

```text
repo_id: local/robomind_tienkung_xsens_smoke
episodes: 1
frames: 20
fps: 30
observation.images.camera_top: (480, 640, 3)
observation.state: 26D
action: 26D
task: pick pipe place plate twice
```

## 成功命令

在 sandbox 外执行，因为 sandbox 中无法访问 NVIDIA driver：

```bash
python scripts/train_smolvla_lora.py \
  --dataset-root /tmp/robomind_xsens_lerobot_smoke \
  --dataset-repo-id local/robomind_tienkung_xsens_smoke \
  --output-dir /tmp/robomind_smolvla_smoke_train_cuda \
  --steps 1 \
  --batch-size 1 \
  --lora-rank 8
```

关键覆盖项：

- `--policy.input_features=null` 和 `--policy.output_features=null`：让 LeRobot 根据当前 dataset 推断 `camera_top`、26D state 和 26D action。否则 `smolvla_base` 默认期待 ALOHA 风格 `camera1/camera2/camera3` 和 6D action。
- `--policy.push_to_hub=false`：避免预训练 config 继承 hub push 设置后要求 `policy.repo_id`。
- `HF_HUB_OFFLINE=1`：使用本地已下载的 `lerobot/smolvla_base` 和 SmolVLM2 权重。
- 清理 proxy 环境变量：当前 `ALL_PROXY=socks://127.0.0.1:7897/` 会导致 `httpx` 报 `Unknown scheme for proxy URL`。

这些覆盖项已固化在 `scripts/train_smolvla_lora.py` 中。脚本默认使用 `lerobot312` 环境，调用：

```text
conda run -n lerobot312 lerobot-train
```

正式训练时建议把转换后的 LeRobotDataset 放到 `/home/slzheng/datasets/...` 下，避免 `/tmp` 被系统清理。

## 结果

CUDA smoke training 成功完成 1 step：

```text
policy.device: cuda
dataset.num_frames: 20
dataset.num_episodes: 1
effective batch size: 1
num_learnable_params: 371328
num_total_params: 450417504
step: 1
loss: 3858297.750
checkpoint: /tmp/robomind_smolvla_smoke_train_cuda/checkpoints/000001
```

checkpoint 主要文件：

```text
pretrained_model/adapter_config.json
pretrained_model/adapter_model.safetensors
pretrained_model/config.json
pretrained_model/policy_preprocessor.json
pretrained_model/policy_postprocessor.json
pretrained_model/train_config.json
training_state/training_step.json
training_state/optimizer_state.safetensors
training_state/scheduler_state.json
```

`training_state/training_step.json` 内容：

```json
{
  "step": 1
}
```

## 注意

当前 checkpoint 是 LoRA adapter smoke artifact，只能说明训练链路能跑通。后续正式 baseline 还需要：

- 使用更多 Xsens episodes。
- 明确训练/验证 split。
- 决定是否继续使用 absolute action，或比较 next-state/delta action。
- 编写 policy inference wrapper，把 SmolVLA 输出拆为 XVLA benchmark 的 `left_arm/right_arm/left_hand/right_hand` payload。
