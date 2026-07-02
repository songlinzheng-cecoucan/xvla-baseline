# 设计：XVLA 相关 Xsens 多任务 Baseline

## 阶段目标

当前阶段目标是从单任务 `pick_pipe_place_plate_twice` baseline 迁移到 XVLA 相关多任务 Xsens baseline。

不改变已有核心数据契约：

```text
observation.images.camera_top
observation.state: 26D
action: 26D
task: language_raw
```

不改变 canonical 26D 映射：

```text
0:7    left_arm
7:14   right_arm
14:20  left_hand
20:26  right_hand
```

## 数据来源

已确认的 Xsens compressed 目录：

```text
benchmark1_0_compressed/h5_tienkung_xsens_1rgb
benchmark1_1_compressed/h5_tienkung_xsens_1rgb
```

完整清单应优先通过登录态工具获取：

```bash
huggingface-cli login
huggingface-cli download x-humanoid-robomind/RoboMIND \
  --repo-type dataset \
  --local-dir /home/slzheng/datasets/RoboMIND \
  --include "benchmark1_0_compressed/h5_tienkung_xsens_1rgb/*"
```

如果只下载候选 task，使用更窄的 include pattern，避免误下完整 Xsens 子目录。

## Task 分类

### 高优先级

| task | 分类 | 对 XVLA 的价值 |
| --- | --- | --- |
| `gear_place` | 齿轮/工业零件放置 | 直接接近“左手把齿轮放到托盘” |
| `place_button` | 按钮/工业开关放置 | 接近“放置工业开关/红色按钮” |
| `place_button_then_press` | 放置按钮 + 按压 | 接近按钮任务，也学习接触/按压 |
| `pick_shelf_insert_machine_press_switch_place_plate` | 取物 + 插入机器 + 按开关 + 放盘 | 包含 switch/press/place 复合技能 |
| `pick_pipe_place_plate_twice` | 小物体抓取并放盘 | 当前已训练，接近托盘/盘子放置 |

### 中高优先级

| task | 分类 | 备注 |
| --- | --- | --- |
| `battery_insertion_with_pullout` | 电池插入/拔出 | 工业精密操作 |
| `plug_insertion` | 插头插入 | 精密插入 |
| `plug_insertion_v2` | 插头插入 v2 | 精密插入 |
| `pick_cup_pour_cup_place` | 抓杯/倾倒/放置 | 泛化抓取放置和腕部姿态 |
| `push_cup_pick_cup_insert_cup` | 推杯 + 抓杯 + 插入 | 含接触和插入 |
| `nut_place` | 螺母放置 | 工业小物体放置 |
| `cylinder_pick_box_place_close` | 圆柱抓取 + 放盒 + 关闭 | 含放置和 close 动作 |

### 暂不优先

| task | 分类 | 原因 |
| --- | --- | --- |
| `plate_push` | 推盘 | 与 XVLA 主任务相关性一般 |
| `plug_extract_from` | 插头拔出 | 可作为反向补充，但不优先 |
| `plug_pullout` | 插头拉出 | 可作为反向补充，但不优先 |
| `brick_piled_then_press_thrice` | 堆叠 + 多次按压 | 接触技能相关，但任务语义较远 |

## XVLA 对应关系

| XVLA 任务 | 首选 Xsens 近邻 | 说明 |
| --- | --- | --- |
| 打开/关闭电子天平侧门 | `cylinder_pick_box_place_close`、`pick_shelf_insert_machine_press_switch_place_plate` | 只提供 close/接触/复合操作近邻 |
| 双手传递开关放进托盘 | `pick_pipe_place_plate_twice`、`place_button`、`pick_shelf_insert_machine_press_switch_place_plate` | 缺少明确 handover |
| 左手把齿轮放到托盘 | `gear_place` | 最高相关 |
| 双手放置工业开关/红色按钮 | `place_button`、`place_button_then_press` | 最高相关 |

## Dataset Manifest

多任务转换后必须写出 task distribution manifest，至少包含：

```json
{
  "source_roots": ["..."],
  "tasks": {
    "gear_place": {
      "num_episodes": 100,
      "num_frames": 40000,
      "source_dir": "benchmark1_0_compressed/h5_tienkung_xsens_1rgb/gear_place"
    }
  },
  "total_episodes": 0,
  "total_frames": 0,
  "canonical_action_dim": 26
}
```

## 训练策略

第一版多任务 baseline 使用同一套 SmolVLA LoRA 配置：

```text
base policy: lerobot/smolvla_base
LoRA rank: 16
batch_size: 1
steps: 10000-50000，随数据规模调整
save_freq: 2000
split: episode-level train/val/test
```

评估必须同时记录：

- global validation/test flow-matching loss。
- per-task validation/test loss。
- open-loop action MAE/MSE。
- 至少每个高优先级 task 一个 held-out episode 视频。

## 判断边界

多任务 Xsens baseline 仍然只是 imitation sanity check：

- 不代表 XVLA benchmark success rate。
- 不代表 TianYi2.0/BrainCo2 动作语义已经对齐。
- 不替代后续 ZMQ policy service 和 Isaac Sim 闭环评测。
