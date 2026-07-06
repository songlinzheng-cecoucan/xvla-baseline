# Xsens 下载记录

## 2026-07-03：place_button

下载方式：

```bash
env \
  -u ALL_PROXY -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY \
  -u http_proxy -u https_proxy \
  HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download x-humanoid-robomind/RoboMIND \
    --repo-type dataset \
    --local-dir /home/slzheng/datasets/RoboMIND \
    --include "benchmark1_0_compressed/h5_tienkung_xsens_1rgb/place_button.tar.gz.part-*" \
    --max-workers 1
```

结果：

```text
download status: success
elapsed: 18m18s
local root: /home/slzheng/datasets/RoboMIND
task path: /home/slzheng/datasets/RoboMIND/benchmark1_0_compressed/h5_tienkung_xsens_1rgb
```

下载文件：

```text
place_button.tar.gz.part-aa  10,737,418,240 bytes
place_button.tar.gz.part-ab  10,737,418,240 bytes
place_button.tar.gz.part-ac   3,659,260,934 bytes
```

合计：

```text
25,134,097,414 bytes
23.41 GiB
du: 24G
```

下载后磁盘：

```text
filesystem: /dev/nvme0n1p3
available: 979G
use: 44%
```

下一步：

- 合并分片为 `place_button.tar.gz`。
- 解压到本地 HDF5 目录。
- 用 Xsens reader 检查至少一个真实 `trajectory.hdf5`。

## 2026-07-03：place_button 解压和验证

解压方式：

```bash
mkdir -p /home/slzheng/datasets/xvla/robomind_xsens_place_button_hdf5
cat /home/slzheng/datasets/RoboMIND/benchmark1_0_compressed/h5_tienkung_xsens_1rgb/place_button.tar.gz.part-* \
  | tar -xzf - -C /home/slzheng/datasets/xvla/robomind_xsens_place_button_hdf5
```

结果：

```text
hdf5 root: /home/slzheng/datasets/xvla/robomind_xsens_place_button_hdf5
task dir: /home/slzheng/datasets/xvla/robomind_xsens_place_button_hdf5/place_button
trajectory.hdf5 count: 278
du: 25G
```

Xsens reader 全量验证：

```text
num_files: 278
failures: 0
total_frames: 108,237
min_frames: 203
max_frames: 996
mean_frames: 389.34
```

样本 task：

```text
pick up the button,place it on the desk,press the button,press the button
```

样本 schema：

```text
state_shape:  (497, 26)
action_shape: (497, 26)
first_rgb_shape: (480, 640, 3)
contains_nan: false
```

## 2026-07-03：place_button LeRobotDataset 转换

转换命令：

```bash
conda run -n lerobot312 python scripts/convert_xsens_to_lerobot.py \
  /home/slzheng/datasets/xvla/robomind_xsens_place_button_hdf5 \
  --output-root /home/slzheng/datasets/xvla/robomind_xsens_place_button_lerobot_278 \
  --repo-id local/robomind_xsens_place_button_278
```

结果：

```text
dataset root: /home/slzheng/datasets/xvla/robomind_xsens_place_button_lerobot_278
repo_id: local/robomind_xsens_place_button_278
episodes: 278
frames: 108,237
fps: 30
use_videos: false
du: 40G
```

LeRobotDataset 加载验证：

```text
len: 108,237
num_episodes: 278
observation.state shape: (26,)
action shape: (26,)
observation.images.camera_top shape: (3, 480, 640)
```

split：

```text
split manifest: /home/slzheng/datasets/xvla/robomind_xsens_place_button_lerobot_278/split_seed1000.json
train/val/test episodes: 222 / 28 / 28
split overlap: 0
```

task distribution manifest：

```text
/home/slzheng/datasets/xvla/robomind_xsens_place_button_lerobot_278/task_distribution_manifest.json
place_button episodes: 278
place_button frames: 108,237
```

## 2026-07-03：pick_pipe + place_button 合并数据集和 smoke 训练

合并转换命令：

```bash
conda run -n lerobot312 python scripts/convert_xsens_to_lerobot.py \
  /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_hdf5_134 \
  /home/slzheng/datasets/xvla/robomind_xsens_place_button_hdf5 \
  --output-root /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_place_button_lerobot_412 \
  --repo-id local/robomind_xsens_pick_pipe_place_button_412
```

结果：

```text
dataset root: /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_place_button_lerobot_412
repo_id: local/robomind_xsens_pick_pipe_place_button_412
episodes: 412
frames: 157,099
du: 60G
```

task distribution：

```text
pick_pipe_place_plate_twice: 134 episodes, 48,862 frames, 小物体放入托盘/盘子
place_button: 278 episodes, 108,237 frames, 按钮/开关操作
```

split：

```text
split manifest: /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_place_button_lerobot_412/split_seed1000.json
train/val/test episodes: 330 / 41 / 41
split overlap: 0
```

LeRobotDataset 加载验证：

```text
len: 157,099
num_episodes: 412
observation.state shape: (26,)
action shape: (26,)
observation.images.camera_top shape: (3, 480, 640)
```

full train split 尝试：

```text
output_dir: /home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_place_button_412ep_2000
episodes: 330
result: interrupted before training step
reason: LeRobot 初始化阶段约 13 分钟内 RSS 上升到约 40GB，仍未进入 GPU step
```

balanced smoke split：

```text
split manifest: /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_place_button_lerobot_412/split_smoke_seed1000_60.json
episodes: 60
pick_pipe_place_plate_twice: 30
place_button: 30
```

smoke 训练：

```bash
python scripts/train_smolvla_lora.py \
  --dataset-root /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_place_button_lerobot_412 \
  --dataset-repo-id local/robomind_xsens_pick_pipe_place_button_412 \
  --output-dir /home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_place_button_60ep_smoke_1000 \
  --split-manifest /home/slzheng/datasets/xvla/robomind_xsens_pick_pipe_place_button_lerobot_412/split_smoke_seed1000_60.json \
  --split-name train \
  --steps 1000 \
  --batch-size 1 \
  --lora-rank 16 \
  --save-freq 500 \
  --log-freq 25 \
  --hf-datasets-cache /tmp/hf_datasets_cache_xvla_smoke60
```

训练结果：

```text
checkpoint: /home/slzheng/datasets/xvla/runs/smolvla_lora_pick_pipe_place_button_60ep_smoke_1000/checkpoints/001000/pretrained_model
dataset.num_frames: 24,787
dataset.num_episodes: 60
num_learnable_params: 742,656
num_total_params: 450,788,832
speed: about 2.27 step/s
step 1000 train loss: 1.056
```

held-out loss：

```text
val max100 mean_loss: 1.0435
test max100 mean_loss: 1.2277
```

## 2026-07-04：gear_place 解压、验证和转换

compressed shards：

```text
source root: /home/slzheng/datasets/RoboMIND/benchmark1_0_compressed/h5_tienkung_xsens_1rgb
gear_place.tar.gz.part-aa ... part-ag
compressed du: 70G
```

解压结果：

```text
hdf5 root: /home/slzheng/datasets/xvla/robomind_xsens_gear_place_hdf5
trajectory.hdf5 count: 507
du: 73G
```

Xsens reader 全量验证：

```text
num_files: 507
failures: 0
total_frames: 309,964
min_frames: 148
max_frames: 3,812
mean_frames: 611.37
task: pick up the big nut,place it into the box,pick up the small nut,place it into the box
image_size: 640x480
```

LeRobotDataset 转换：

```text
dataset root: /home/slzheng/datasets/xvla/robomind_xsens_gear_place_lerobot_507
repo_id: local/robomind_xsens_gear_place_507
episodes: 507
frames: 309,964
fps: 30
du: 120G
```

## 2026-07-04：4-task balanced v3 数据集和训练

balanced 数据集构建工具：

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

转换结果：

```text
dataset root: /home/slzheng/datasets/xvla/robomind_xsens_xvla_multitask_v3_balanced_lerobot_120
repo_id: local/robomind_xsens_xvla_multitask_v3_balanced_120
episodes: 120
frames: 67,997
tasks: 4
du: 25G
selected manifest: selected_hdf5_manifest.json
```

task distribution：

```text
gear_place: 30 episodes, 19,294 frames, 齿轮/工业零件放置
pick_pipe_place_plate_twice: 30 episodes, 10,520 frames, 小物体放入托盘/盘子
pick_shelf_insert_machine_press_switch_place_plate: 30 episodes, 26,760 frames, 按钮/开关操作+小物体放入托盘/盘子
place_button: 30 episodes, 11,423 frames, 按钮/开关操作
```

split：

```text
split manifest: /home/slzheng/datasets/xvla/robomind_xsens_xvla_multitask_v3_balanced_lerobot_120/split_seed1000_stratified.json
split type: episode_stratified_by_task
train/val/test episodes: 96 / 12 / 12
per-task train/val/test episodes: 24 / 3 / 3
```

LeRobotDataset 加载验证：

```text
len: 67,997
num_episodes: 120
fps: 30
observation.state shape: (26,)
action shape: (26,)
observation.images.camera_top shape: (3, 480, 640)
```

SmolVLA LoRA 训练：

```bash
python scripts/train_smolvla_lora.py \
  --dataset-root /home/slzheng/datasets/xvla/robomind_xsens_xvla_multitask_v3_balanced_lerobot_120 \
  --dataset-repo-id local/robomind_xsens_xvla_multitask_v3_balanced_120 \
  --output-dir /home/slzheng/datasets/xvla/runs/smolvla_lora_xvla_multitask_v3_balanced120_5000 \
  --split-manifest /home/slzheng/datasets/xvla/robomind_xsens_xvla_multitask_v3_balanced_lerobot_120/split_seed1000_stratified.json \
  --split-name train \
  --steps 5000 \
  --batch-size 1 \
  --lora-rank 16 \
  --save-freq 1000 \
  --log-freq 25 \
  --eval-freq 0 \
  --num-workers 0 \
  --hf-datasets-cache /tmp/hf_datasets_cache_xvla_v3_balanced120
```

训练结果：

```text
output_dir: /home/slzheng/datasets/xvla/runs/smolvla_lora_xvla_multitask_v3_balanced120_5000
train episodes: 96
train frames: 53,754
steps: 5,000
checkpoints: 001000, 002000, 003000, 004000, 005000
final checkpoint: checkpoints/005000/pretrained_model
num_learnable_params: 742,656
num_total_params: 450,788,832
initial step-25 loss: 4.012
step-1000 loss: 1.200
step-3000 recent loss range: about 0.72-1.09
final step-5000 loss: 0.625
training log: /tmp/xvla_v3_smolvla_train.log
```
