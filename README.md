# XVLA Baseline

This repository contains baseline data, training, and offline evaluation tooling for the XVLA VLA operation track.

Current method: adapt RoboMIND TienKung Xsens trajectories to a 26D action interface and fine-tune `lerobot/smolvla_base` with LoRA. The repository is structured so additional policy families can be added later without changing the project identity.

The repository intentionally does not include raw HDF5 files, LeRobot datasets, checkpoints, TensorBoard logs, or generated videos.

## Scope

Implemented:

- Read and validate RoboMIND TienKung Xsens HDF5 trajectories.
- Map Xsens `puppet/joint_position` and `puppet/end_effector` to a canonical XVLA 26D vector.
- Convert selected trajectories to a LeRobotDataset.
- Create episode-level train/val/test split manifests.
- Launch SmolVLA LoRA training in `lerobot312`.
- Evaluate checkpoint flow-matching loss on held-out episodes.
- Inspect open-loop action predictions with JSON/CSV/PNG outputs.
- Render open-loop action comparison videos.
- Export existing JSON metrics to TensorBoard event files.

Not implemented yet:

- XVLA benchmark ZMQ policy service.
- Closed-loop Isaac Sim rollout.
- TianYi2.0/BrainCo2-specific action remapping or safety control.
- Full training runner preflight/manifest implementation.

## Environment Split

Keep simulation and training environments separate:

```text
xmimic      Isaac Sim / XVLA benchmark runtime
lerobot312  LeRobot dataset conversion, SmolVLA training, evaluation
```

The scripts in this repository assume `lerobot312` has LeRobot, PyTorch, PEFT, Transformers, Pillow, OpenCV, ImageIO, and TensorBoard available.

Helper dependencies used directly by this repository are listed in `requirements.txt`. LeRobot, PyTorch, Transformers, and PEFT should be installed according to the training environment you use.

## Repository Layout

```text
src/xvla_baseline/
  data/       RoboMIND Xsens reader, LeRobot conversion, episode split helpers
  training/   SmolVLA LoRA training launcher
  eval/       loss evaluation, open-loop inspection, video rendering, TensorBoard export
scripts/      thin CLI wrappers for the package modules
openspec/     spec-first design and task tracking
robomind_static/static/
              small RoboMIND schema/reference files
```

For quick local use, run the wrappers in `scripts/`. For installed usage, run:

```bash
pip install -e .
```

Then use the console commands declared in `pyproject.toml`, such as `xvla-eval-loss` or `xvla-inspect-open-loop`.

## Data Layout

Local datasets and training artifacts should live outside the git repository:

```text
/home/slzheng/datasets/xvla
```

Current local baseline artifacts used during development:

```text
/home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_hdf5_134
/home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_lerobot_134
/home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_134ep_5000
```

These paths are examples from the local machine; teammates should adjust them for their own data location.

## Canonical 26D Action

The baseline uses one fixed 26D state/action vector:

```text
0:7    left_arm
7:14   right_arm
14:20  left_hand
20:26  right_hand
```

RoboMIND TienKung Xsens mapping:

```text
puppet/joint_position[:, 0:7]   -> left_arm
puppet/joint_position[:, 7:14]  -> right_arm
puppet/end_effector[:, 0:6]     -> left_hand
puppet/end_effector[:, 6:12]    -> right_hand
```

Reader v1 uses absolute target semantics:

```text
state[t] = puppet canonical26[t]
action[t] = puppet canonical26[t]
```

This is a baseline imitation-learning convention, not a final benchmark control contract.

## Main Scripts

### Convert Xsens HDF5 to LeRobotDataset

```bash
conda run -n lerobot312 python scripts/convert_xsens_to_lerobot.py \
  /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_hdf5_134 \
  --output-root /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_lerobot_134 \
  --repo-id local/robomind_xsens_pick_pipe_134 \
  --overwrite
```

### Create Episode Split

```bash
python scripts/split_lerobot_episodes.py \
  --dataset-root /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_lerobot_134 \
  --dataset-repo-id local/robomind_xsens_pick_pipe_134 \
  --output /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_lerobot_134/split_seed1000.json \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 1000
```

### Train SmolVLA LoRA

```bash
python scripts/train_smolvla_lora.py \
  --dataset-root /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_lerobot_134 \
  --dataset-repo-id local/robomind_xsens_pick_pipe_134 \
  --output-dir /home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_134ep_5000 \
  --split-manifest /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_lerobot_134/split_seed1000.json \
  --split-name train \
  --steps 5000 \
  --batch-size 1 \
  --lora-rank 16 \
  --save-freq 1000 \
  --log-freq 50
```

### Evaluate Flow-Matching Loss

```bash
env \
  -u ALL_PROXY -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY \
  -u http_proxy -u https_proxy \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_CACHE=/tmp/hf_datasets_cache \
  conda run -n lerobot312 python scripts/eval_smolvla_val_loss.py \
    --checkpoint /home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_134ep_5000/checkpoints/005000/pretrained_model \
    --dataset-root /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_lerobot_134 \
    --dataset-repo-id local/robomind_xsens_pick_pipe_134 \
    --split-manifest /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_lerobot_134/split_seed1000.json \
    --split-name test \
    --batch-size 1 \
    --device cuda \
    --max-batches 100 \
    --output /home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_134ep_5000/test_loss_005000_max100.json
```

### Inspect Open-Loop Actions

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
    --device cuda \
    --output-dir /home/slzheng/datasets/xvla/reports/open_loop_005000_test
```

### Render Open-Loop Video

```bash
env \
  -u ALL_PROXY -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY \
  -u http_proxy -u https_proxy \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_CACHE=/tmp/hf_datasets_cache \
  conda run -n lerobot312 python scripts/render_smolvla_open_loop_video.py \
    --checkpoint /home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_134ep_5000/checkpoints/005000/pretrained_model \
    --dataset-root /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_lerobot_134 \
    --dataset-repo-id local/robomind_xsens_pick_pipe_134 \
    --episode 16 \
    --output /home/slzheng/datasets/xvla/reports/open_loop_005000_test_episode16/open_loop_episode016.mp4 \
    --max-frames 80 \
    --stride 2 \
    --fps 10 \
    --device cuda
```

## Baseline Result Snapshot

The current local 134-episode run used:

```text
train/val/test episodes: 107 / 13 / 14
frames: 48,862
steps: 5,000
batch_size: 1
LoRA rank: 16
```

Checkpoint-level smoke metrics:

```text
val 001000 flow-matching loss max100: 1.3406
val 003000 flow-matching loss max100: 0.7882
val 005000 flow-matching loss max100: 0.7523
test 005000 flow-matching loss max100: 0.7973
```

These are open-loop/offline imitation metrics. They do not imply XVLA closed-loop benchmark success.

## TensorBoard

The LeRobot training run used during development did not emit native TensorBoard logs. Existing JSON metrics can be exported:

```bash
conda run -n lerobot312 python scripts/export_metrics_to_tensorboard.py \
  --run-dir /home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_134ep_5000 \
  --log-dir /home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_134ep_5000/tensorboard
```

Start TensorBoard:

```bash
conda run -n lerobot312 tensorboard \
  --logdir /home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_134ep_5000/tensorboard \
  --host 127.0.0.1 \
  --port 6006
```

## OpenSpec

The active baseline design is tracked under:

```text
openspec/changes/add-xvla-smolvla-baseline
```

Validate the active change:

```bash
openspec validate add-xvla-smolvla-baseline
```

## Repository Hygiene

Do not commit:

- raw HDF5 files
- LeRobotDataset exports
- checkpoints
- TensorBoard logs
- generated reports
- generated videos
- Hugging Face cache files
