# XVLA Baseline

本仓库提供 XVLA VLA 操作赛道的 baseline 数据处理、训练和离线评估工具。

当前方法：将 RoboMIND TienKung Xsens 轨迹适配到 XVLA 26 维动作接口，并基于 `lerobot/smolvla_base` 做 LoRA 微调。仓库结构刻意保持通用，后续可以继续加入 OpenVLA、pi0、ACT、Diffusion Policy 或自研策略，而不需要更换项目名称。

本仓库不包含原始 HDF5、LeRobotDataset、checkpoint、TensorBoard 日志或生成视频。

## 范围

已实现：

- 读取并校验 RoboMIND TienKung Xsens HDF5 轨迹。
- 将 Xsens 的 `puppet/joint_position` 和 `puppet/end_effector` 映射到 XVLA canonical 26D 向量。
- 将选定轨迹转换为 LeRobotDataset。
- 生成 episode-level train / val / test split manifest。
- 在 `lerobot312` 环境中启动 SmolVLA LoRA 训练。
- 在 held-out episodes 上评估 checkpoint 的 flow-matching loss。
- 做 open-loop action inspection，输出 JSON / CSV / PNG。
- 渲染 open-loop action 对比视频。
- 将已有 JSON 指标导出为 TensorBoard event 文件。

暂未实现：

- XVLA benchmark ZMQ policy service。
- Isaac Sim 闭环 rollout。
- TianYi2.0 / BrainCo2 专属动作 remapping 或安全控制。
- 完整 train runner 的 preflight / manifest / failure classification。

## 环境分工

仿真和训练环境保持分离：

```text
xmimic      Isaac Sim / XVLA benchmark runtime
lerobot312  LeRobot 数据转换、SmolVLA 训练、离线评估
```

本仓库脚本默认 `lerobot312` 中已有 LeRobot、PyTorch、PEFT、Transformers、Pillow、OpenCV、ImageIO 和 TensorBoard。

`requirements.txt` 只列出本仓库脚本直接使用的辅助依赖。LeRobot、PyTorch、Transformers 和 PEFT 请按你的训练环境单独安装。

## 仓库结构

```text
src/xvla_baseline/
  data/       RoboMIND Xsens reader、LeRobot 转换、episode split 工具
  training/   SmolVLA LoRA 训练 launcher
  eval/       loss 评估、open-loop inspection、视频渲染、TensorBoard 导出
scripts/      package module 的轻量 CLI wrapper
openspec/     spec-first 设计和任务管理
robomind_static/static/
              小型 RoboMIND schema / reference 文件
```

本地快速使用时，直接运行 `scripts/` 下的 wrapper。若希望安装成包：

```bash
pip install -e .
```

安装后可以使用 `pyproject.toml` 中声明的 console commands，例如：

```text
xvla-task-distribution-manifest
xvla-eval-loss
xvla-inspect-open-loop
xvla-render-open-loop-video
```

## 数据目录

本地数据和训练产物应放在 git 仓库外：

```text
/home/slzheng/datasets/xvla
```

当前开发时使用过的本地产物示例：

```text
/home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_hdf5_134
/home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_lerobot_134
/home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_place_button_lerobot_412
/home/slzheng/datasets/xvla/robomind_xsens_xvla_multitask_v3_balanced_lerobot_120
/home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_134ep_5000
/home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_place_button_60ep_smoke_1000
/home/slzheng/datasets/xvla/runs/smolvla_lora_xvla_multitask_v3_balanced120_5000
```

这些路径只是本机示例。队友使用时应替换成自己的数据目录。

## Canonical 26D 动作

baseline 使用固定的 26 维 state / action 向量：

```text
0:7    left_arm
7:14   right_arm
14:20  left_hand
20:26  right_hand
```

RoboMIND TienKung Xsens 到 canonical 26D 的映射：

```text
puppet/joint_position[:, 0:7]   -> left_arm
puppet/joint_position[:, 7:14]  -> right_arm
puppet/end_effector[:, 0:6]     -> left_hand
puppet/end_effector[:, 6:12]    -> right_hand
```

Reader v1 使用 absolute target 语义：

```text
state[t] = puppet canonical26[t]
action[t] = puppet canonical26[t]
```

这是 baseline 阶段的 imitation-learning 约定，不是最终 benchmark 控制接口契约。

## 主要命令

### 转换 Xsens HDF5 为 LeRobotDataset

```bash
conda run -n lerobot312 python scripts/convert_xsens_to_lerobot.py \
  /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_hdf5_134 \
  --output-root /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_lerobot_134 \
  --repo-id local/robomind_xsens_pick_pipe_134 \
  --overwrite
```

### 生成 episode split

```bash
python scripts/split_lerobot_episodes.py \
  --dataset-root /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_lerobot_134 \
  --dataset-repo-id local/robomind_xsens_pick_pipe_134 \
  --output /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_lerobot_134/split_seed1000.json \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 1000
```

若已经有 task distribution manifest，可以生成按任务分层的 split：

```bash
python scripts/split_lerobot_episodes.py \
  --dataset-root /home/slzheng/datasets/xvla/robomind_xsens_xvla_multitask_v3_balanced_lerobot_120 \
  --dataset-repo-id local/robomind_xsens_xvla_multitask_v3_balanced_120 \
  --output /home/slzheng/datasets/xvla/robomind_xsens_xvla_multitask_v3_balanced_lerobot_120/split_seed1000_stratified.json \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 1000 \
  --task-distribution-manifest /home/slzheng/datasets/xvla/robomind_xsens_xvla_multitask_v3_balanced_lerobot_120/task_distribution_manifest.json
```

### 生成 task distribution manifest

```bash
python scripts/write_task_distribution_manifest.py \
  --dataset-root /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_place_button_lerobot_412 \
  --dataset-repo-id local/robomind_xsens_pick_pipe_place_button_412 \
  --source-root /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_hdf5_134 \
  --source-root /home/slzheng/datasets/xvla/robomind_xsens_place_button_hdf5 \
  --task-category 'pick_pipe_place_plate_twice=小物体放入托盘/盘子' \
  --task-category 'place_button=按钮/开关操作' \
  --output /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_place_button_lerobot_412/task_distribution_manifest.json
```

### 构建 balanced 多任务数据集

```bash
python scripts/build_balanced_xsens_lerobot.py \
  --task-root pick_pipe_place_plate_twice=/home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_hdf5_134 \
  --task-root place_button=/home/slzheng/datasets/xvla/robomind_xsens_place_button_hdf5 \
  --task-root pick_shelf_insert_machine_press_switch_place_plate=/home/slzheng/datasets/xvla/robomind_xsens_pick_shelf_hdf5 \
  --task-root gear_place=/home/slzheng/datasets/xvla/robomind_xsens_gear_place_hdf5 \
  --episodes-per-task 30 \
  --seed 1000 \
  --output-root /home/slzheng/datasets/xvla/robomind_xsens_xvla_multitask_v3_balanced_lerobot_120 \
  --repo-id local/robomind_xsens_xvla_multitask_v3_balanced_120 \
  --overwrite
```

### 训练 SmolVLA LoRA

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

### 评估 flow-matching loss

```bash
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

### Open-loop action inspection

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

### 渲染 open-loop 对比视频

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

## 当前 baseline 结果快照

当前本机 134-episode run 使用：

```text
train/val/test episodes: 107 / 13 / 14
frames: 48,862
steps: 5,000
batch_size: 1
LoRA rank: 16
```

checkpoint 级别 smoke 指标：

```text
val 001000 flow-matching loss max100: 1.3406
val 003000 flow-matching loss max100: 0.7882
val 005000 flow-matching loss max100: 0.7523
test 005000 flow-matching loss max100: 0.7973
```

多任务 smoke run：

```text
dataset: local/robomind_xsens_pick_pipe_place_button_412
dataset root: /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_place_button_lerobot_412
episodes: 412
frames: 157,099
tasks:
  pick_pipe_place_plate_twice: 134 episodes / 48,862 frames
  place_button: 278 episodes / 108,237 frames
smoke train split: 60 episodes, balanced 30 + 30
steps: 1,000
batch_size: 1
LoRA rank: 16
checkpoint: /home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_place_button_60ep_smoke_1000/checkpoints/001000/pretrained_model
val flow-matching loss max100: 1.0435
test flow-matching loss max100: 1.2277
```

full 330-episode train split 曾尝试直接启动，但 LeRobot 初始化阶段 13 分钟内 RSS 上升到约 40GB 且尚未进入 GPU step；当前建议先用 balanced smoke split 验证，再优化 full-split 训练入口。

4-task balanced v3 run：

```text
dataset: local/robomind_xsens_xvla_multitask_v3_balanced_120
dataset root: /home/slzheng/datasets/xvla/robomind_xsens_xvla_multitask_v3_balanced_lerobot_120
episodes: 120
frames: 67,997
tasks:
  gear_place: 30 episodes / 19,294 frames
  pick_pipe_place_plate_twice: 30 episodes / 10,520 frames
  pick_shelf_insert_machine_press_switch_place_plate: 30 episodes / 26,760 frames
  place_button: 30 episodes / 11,423 frames
stratified train/val/test episodes: 96 / 12 / 12
per-task train/val/test episodes: 24 / 3 / 3
steps: 5,000
batch_size: 1
LoRA rank: 16
checkpoint: /home/slzheng/datasets/xvla/runs/smolvla_lora_xvla_multitask_v3_balanced120_5000/checkpoints/005000/pretrained_model
final train loss: 0.625
```

4-task balanced232 formal run：

```text
dataset: local/robomind_xsens_xvla_multitask_v4_balanced_232
dataset root: /home/slzheng/datasets/xvla/robomind_xsens_xvla_multitask_v4_balanced_lerobot_232
episodes: 232
frames: 129,186
tasks:
  gear_place: 58 episodes / 32,170 frames
  pick_pipe_place_plate_twice: 58 episodes / 20,670 frames
  pick_shelf_insert_machine_press_switch_place_plate: 58 episodes / 54,128 frames
  place_button: 58 episodes / 22,218 frames
stratified train/val/test episodes: 184 / 24 / 24
per-task train/val/test episodes: 46 / 6 / 6
steps: 50,000
batch_size: 2
num_workers: 2
LoRA rank: 16
checkpoint: /home/slzheng/datasets/xvla/runs/smolvla_lora_xvla_multitask_v4_balanced232_50000_bs2_nw2/checkpoints/050000/pretrained_model
```

held-out flow-matching loss：

```text
validation global: 0.2771
  gear_place: 0.3174
  pick_pipe_place_plate_twice: 0.2253
  pick_shelf_insert_machine_press_switch_place_plate: 0.2204
  place_button: 0.3511
test global: 0.2863
  gear_place: 0.4755
  pick_pipe_place_plate_twice: 0.2053
  pick_shelf_insert_machine_press_switch_place_plate: 0.2030
  place_button: 0.3621
reports: /home/slzheng/datasets/xvla/eval_reports/balanced232_050000
open-loop videos: /home/slzheng/datasets/xvla/open_loop_videos/balanced232_050000
open-loop inspection: /home/slzheng/datasets/xvla/open_loop_inspections/balanced232_050000_test_200
```

这些是 open-loop / offline imitation 指标，不代表 XVLA 闭环 benchmark 成功率。

## TensorBoard

开发时使用的 LeRobot 训练命令没有直接写出原生 TensorBoard log。已有 JSON 指标可以导出为 TensorBoard event：

```bash
conda run -n lerobot312 python scripts/export_metrics_to_tensorboard.py \
  --run-dir /home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_134ep_5000 \
  --log-dir /home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_134ep_5000/tensorboard
```

启动 TensorBoard：

```bash
conda run -n lerobot312 tensorboard \
  --logdir /home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_134ep_5000/tensorboard \
  --host 127.0.0.1 \
  --port 6006
```

## OpenSpec

当前 baseline 设计记录在：

```text
openspec/changes/add-xvla-xsens-multitask-baseline
```

校验 active change：

```bash
openspec validate add-xvla-xsens-multitask-baseline
```

## 仓库卫生

不要提交：

- 原始 HDF5 文件
- LeRobotDataset 导出
- checkpoint
- TensorBoard 日志
- 生成视频
- Hugging Face cache 文件

可以提交：

- `docs/experiments/` 下的小型 Markdown / PNG / CSV 实验报告
- 复现报告所需的轻量脚本
