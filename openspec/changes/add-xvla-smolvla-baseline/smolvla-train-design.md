# SmolVLA Train 任务设计

## 当前问题

当前 `scripts/train_smolvla_lora.py` 只是一个命令行 wrapper：

```text
parse args
-> 拼接 lerobot-train CLI 参数
-> conda run -n lerobot312 lerobot-train ...
```

它适合记录和复现 smoke command，但还不是一个可维护的训练任务实现。主要不足：

- 没有仓库内的 `train()` 函数，不能被测试、复用或由后续 pipeline 调用。
- 训练前校验很薄，只检查 dataset root 和几个数值参数。
- 不能结构化记录输入数据、feature schema、训练参数、checkpoint 路径和失败类型。
- 不能在训练前主动检查 SmolVLA feature 覆盖项是否正确，例如 `input_features=null`、`output_features=null`。
- 不能区分失败属于 dataset、model loading、PEFT config、CUDA/OOM、proxy/HF cache、还是 LeRobot 内部训练错误。

因此下一版目标不是重写 LeRobot 训练算法，而是提供一个 XVLA baseline 专用的 train runner。

## 设计目标

train runner 必须做到：

- 使用本地 `lerobot312` 环境运行。
- 使用 `lerobot/smolvla_base` 作为默认 base policy。
- 使用 LoRA/PEFT，默认不做 full fine-tuning。
- 支持 RoboMIND Xsens 转换后的 LeRobotDataset：

```text
observation.images.camera_top
observation.state: 26D
action: 26D
task
```

- 显式覆盖 SmolVLA 预训练 ALOHA feature schema：

```text
policy.input_features = None
policy.output_features = None
policy.push_to_hub = False
```

- 在训练前执行 preflight validation。
- 训练后记录 checkpoint manifest。
- 对常见失败给出结构化分类，便于排查。

train runner 不得：

- 自己重写 SmolVLA forward、loss、optimizer step 或 PEFT wrapping。
- 静默把 26D action 改成 ALOHA 6D。
- 静默把 `camera_top` 改名成 `camera1/camera2/camera3`。
- 把大型 checkpoint 或 dataset 写进 git。

## 模块边界

推荐保留两层入口：

```text
scripts/train_smolvla_lora.py                    # thin CLI wrapper
src/xvla_baseline/training/train_smolvla_lora.py # train launcher implementation
```

### `scripts/train_smolvla_lora.py`

职责：

- 解析用户命令行。
- 可选清理 proxy 环境变量。
- 设置 HF cache / offline / wandb 环境变量。
- 调用：

```bash
conda run -n lerobot312 python xvla_smolvla_train.py ...
```

它不负责：

- import LeRobot。
- 构造 `TrainPipelineConfig`。
- 加载 dataset 或 policy。
- 解释训练失败。

### `xvla_smolvla_train.py`

职责：

- 定义 `XVlaSmolVLATrainConfig`。
- 执行 preflight validation。
- 构造 LeRobot training config。
- 调用 LeRobot 训练入口。
- 写出训练 manifest。
- 捕获并分类常见异常。

建议函数：

```python
@dataclass
class XVlaSmolVLATrainConfig:
    dataset_root: Path
    dataset_repo_id: str
    output_dir: Path
    policy_path: str = "lerobot/smolvla_base"
    steps: int = 1000
    batch_size: int = 1
    lora_rank: int = 8
    save_freq: int = 500
    log_freq: int = 20
    eval_freq: int = 0
    num_workers: int = 0
    offline: bool = True
    clear_proxy: bool = True
```

```python
def preflight_check(cfg: XVlaSmolVLATrainConfig) -> PreflightReport: ...
def build_lerobot_cli_args(cfg: XVlaSmolVLATrainConfig) -> list[str]: ...
def run_lerobot_train(cfg: XVlaSmolVLATrainConfig) -> TrainResult: ...
def write_train_manifest(cfg, preflight, result) -> Path: ...
def classify_training_error(exc: BaseException) -> TrainingFailure: ...
def train(cfg: XVlaSmolVLATrainConfig) -> TrainResult: ...
```

第一版可以继续通过 `lerobot-train` subprocess 运行，避免和 LeRobot 的 `draccus` config parser 过早耦合。关键是把 subprocess 包在可测试的 `train()` 里，并把 preflight、manifest、失败分类做成仓库内逻辑。

后续如果需要更细控制，再从 subprocess 切换为直接 import：

```python
from lerobot.scripts.lerobot_train import train
```

但这一步需要确认 LeRobot 的 `TrainPipelineConfig` 构造方式在当前版本稳定。

## Preflight Validation

训练启动前必须检查：

### Dataset

- `dataset_root` 存在。
- `meta/info.json` 存在。
- `total_frames > 0`。
- `total_episodes > 0`。
- features 包含：

```text
observation.images.camera_top
observation.state shape=[26]
action shape=[26]
```

- `conversion_manifest.json` 存在时，读取并记录 source HDF5 路径、任务名、frame 数。

### Environment

- 当前执行环境能调用 `conda run -n lerobot312`，或 runner 已经在 `lerobot312` 内。
- `torch.cuda.is_available()` 在非 dry-run 训练中应为 true；如果为 false，必须明确标记为 `cuda_unavailable`，不能当作普通训练失败。
- HF cache 中能找到 `lerobot/smolvla_base`。offline 模式下如果缺权重，直接失败并标记 `model_cache_missing`。
- proxy 变量如果包含 `socks://`，默认清理；如果用户指定 `--keep-proxy`，manifest 必须记录。

### Config

- `steps > 0`。
- `batch_size > 0`。
- `lora_rank > 0`。
- `save_freq > 0`。
- 必须写入以下 LeRobot 覆盖项：

```text
--policy.push_to_hub=false
--policy.input_features=null
--policy.output_features=null
--peft.method_type=LORA
--peft.r=<rank>
```

## 训练执行

第一版训练仍复用 LeRobot CLI：

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.input_features=null \
  --policy.output_features=null \
  --dataset.repo_id=<repo_id> \
  --dataset.root=<dataset_root> \
  --output_dir=<output_dir> \
  --batch_size=<batch_size> \
  --steps=<steps> \
  --save_freq=<save_freq> \
  --log_freq=<log_freq> \
  --eval_freq=0 \
  --num_workers=<num_workers> \
  --peft.method_type=LORA \
  --peft.r=<rank> \
  --wandb.enable=false
```

理由：

- LeRobot 已经负责 SmolVLA model creation、processor、normalizer、PEFT wrapping、optimizer、scheduler、checkpoint 保存。
- 我们当前要控制的是 XVLA 数据和训练配置，不是替换 LeRobot 训练内核。

## 输出 Artifact

训练成功后，runner 必须记录：

```text
<output_dir>/xvla_train_manifest.json
```

manifest 至少包含：

```json
{
  "status": "success",
  "dataset_root": "...",
  "dataset_repo_id": "...",
  "policy_path": "lerobot/smolvla_base",
  "steps": 1000,
  "batch_size": 1,
  "lora_rank": 8,
  "feature_schema": {
    "observation.images.camera_top": [480, 640, 3],
    "observation.state": [26],
    "action": [26]
  },
  "checkpoint_dir": ".../checkpoints/001000",
  "pretrained_model_dir": ".../checkpoints/001000/pretrained_model",
  "training_step": 1000
}
```

训练失败时也必须写 manifest：

```json
{
  "status": "failed",
  "failure_type": "cuda_oom",
  "message": "...",
  "command": [...]
}
```

## 失败分类

第一版至少区分：

```text
dataset_missing
dataset_schema_error
model_cache_missing
proxy_error
cuda_unavailable
cuda_oom
feature_mismatch
peft_config_error
checkpoint_missing
lerobot_train_error
unknown
```

已知错误映射：

- `Unknown scheme for proxy URL` -> `proxy_error`
- `Device 'cuda' is not available` 或 `torch.cuda.is_available() == False` -> `cuda_unavailable`
- `Feature mismatch between dataset/environment and policy config` -> `feature_mismatch`
- `No pretrained model name found` / HF offline 缺文件 -> `model_cache_missing`
- `CUDA out of memory` -> `cuda_oom`

## 验收标准

### Smoke

给定 20 帧 tiny dataset：

```bash
python scripts/train_smolvla_lora.py \
  --dataset-root /tmp/robomind_xsens_lerobot_smoke \
  --dataset-repo-id local/robomind_tienkung_xsens_smoke \
  --output-dir /tmp/robomind_smolvla_smoke_train_cuda \
  --steps 1 \
  --batch-size 1 \
  --lora-rank 8
```

必须：

- 在 CUDA 上启动训练。
- 完成至少 1 step。
- 写出 LeRobot checkpoint。
- 写出 `xvla_train_manifest.json`。
- manifest 指向真实存在的 `pretrained_model` 目录。

### Small Formal Run

给定持久化在 `/home/slzheng/datasets/...` 的多 episode dataset：

```bash
python scripts/train_smolvla_lora.py \
  --dataset-root /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_lerobot \
  --dataset-repo-id local/robomind_xsens_pick_pipe \
  --output-dir /home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_001 \
  --steps 500 \
  --batch-size 1 \
  --lora-rank 8 \
  --save-freq 100
```

必须：

- 完成 500 steps 或给出结构化失败原因。
- 每 `save_freq` 产生 checkpoint。
- 训练输出不写入 git。

## 和推理任务的关系

训练 runner 完成后，下一步不是继续加训练功能，而是实现 checkpoint 推理 IO：

```text
pretrained_model checkpoint
-> load policy + processors
-> 输入 camera_top + 26D state + task
-> 输出 26D action
```

只有推理 IO 跑通后，长训练 checkpoint 才能合理接入 XVLA ZMQ policy service。
