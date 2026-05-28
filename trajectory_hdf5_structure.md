# RoboMIND Franka Trajectory HDF5 结构说明

本文档基于下面这个已下载的样例文件整理：

```text
/home/slzheng/datasets/RoboMIND2.0-Franka-Part-3/data/franka/place_coke_bottle_on_tray/success_episodes/0416_130751/data/trajectory.hdf5
```

该文件大小约 `368.72 MiB`，包含一个长度为 `213` 的 trajectory。这里的 `213` 表示 213 个时间步。HDF5 文件中不是直接保存 213 个 Python 字典，而是用一组并行 dataset 保存；所有主要时序数据的第一维都是 `213`，第 `i` 个 sample 需要从各个 dataset 中取第 `i` 项组合出来。

## 顶层元信息

```text
/
├── metadata/                                      # trajectory 级别元信息，保存在 group attributes 中
│   ├── collection_time                            # 采集时间: 2025-05-28 19:00:40
│   ├── collector                                  # 采集者/采集账号 ID: open_a7b36bad82171cc7cd4814bdcf66d40
│   ├── data_type                                  # 数据类型: real，表示真实机器人数据
│   ├── language_instruction                       # 语言任务指令: pick_up_coke_bottle_place_on_plate
│   └── trajectory_length                          # trajectory 长度: 213
```

## 相机配置

```text
/
├── camera_model/                                  # 每个相机的型号
│   ├── camera_front                               # 前视相机型号: RealSense_D435if
│   ├── camera_left                                # 左侧相机型号: RealSense_D435if
│   ├── camera_right                               # 右侧相机型号: RealSense_D435if
│   ├── camera_top                                 # 顶视相机型号: RealSense_D435if
│   ├── camera_wrist_left                          # 左腕部相机型号: RealSense_D435if
│   └── camera_wrist_right                         # 右腕部相机型号: RealSense_D435if
├── camera_color_channel/                          # 彩色图像通道格式
│   ├── camera_front                               # rgb
│   ├── camera_left                                # rgb
│   ├── camera_right                               # rgb
│   ├── camera_top                                 # rgb
│   ├── camera_wrist_left                          # rgb
│   └── camera_wrist_right                         # rgb
├── camera_color_resolution/                       # 彩色图像分辨率，格式为 [height, width]
│   ├── camera_front                               # [720, 1280]
│   ├── camera_left                                # [480, 640]
│   ├── camera_right                               # [480, 640]
│   ├── camera_top                                 # [720, 1280]
│   ├── camera_wrist_left                          # [480, 640]
│   └── camera_wrist_right                         # [480, 640]
└── camera_depth_resolution/                       # 深度图像分辨率，格式为 [height, width]
    ├── camera_front                               # [720, 1280]
    ├── camera_left                                # [480, 640]
    ├── camera_right                               # [480, 640]
    ├── camera_top                                 # [720, 1280]
    ├── camera_wrist_left                          # [480, 640]
    └── camera_wrist_right                         # [480, 640]
```

## 相机观测数据

```text
/
└── camera_observations/                           # 按时间步对齐的相机观测
    ├── timestamp                                  # shape=(213,), dtype=int64；每一帧相机观测的时间戳
    ├── is_intervene                               # shape=(213,), dtype=bool；该时间步是否有人工介入标记
    ├── color_images/                              # 彩色图像，第一维为时间步
    │   ├── camera_front                           # shape=(213,), dtype=object；每项是 JPEG 编码的 uint8 bytes
    │   ├── camera_left                            # shape=(213,), dtype=object；每项是 JPEG 编码的 uint8 bytes
    │   ├── camera_right                           # shape=(213,), dtype=object；每项是 JPEG 编码的 uint8 bytes
    │   ├── camera_top                             # shape=(213,), dtype=object；每项是 JPEG 编码的 uint8 bytes
    │   ├── camera_wrist_left                      # shape=(213,), dtype=object；每项是 JPEG 编码的 uint8 bytes
    │   └── camera_wrist_right                     # shape=(213,), dtype=object；每项是 JPEG 编码的 uint8 bytes
    └── depth_images/                              # 深度图像，第一维为时间步
        ├── camera_front                           # shape=(213,), dtype=object；每项是 PNG 编码的 uint8 bytes
        ├── camera_left                            # shape=(213,), dtype=object；每项是 PNG 编码的 uint8 bytes
        ├── camera_right                           # shape=(213,), dtype=object；每项是 PNG 编码的 uint8 bytes
        ├── camera_top                             # shape=(213,), dtype=object；每项是 PNG 编码的 uint8 bytes
        ├── camera_wrist_left                      # shape=(213,), dtype=object；每项是 PNG 编码的 uint8 bytes
        └── camera_wrist_right                     # shape=(213,), dtype=object；每项是 PNG 编码的 uint8 bytes
```

## 机器人状态和控制数据

`master` 和 `puppet` 各自包含相同的 6 类机器人数据。根据遥操作数据的常见命名和该文件中的数值关系，可以理解为：

- `master`: 遥操作主端数据，即人类操作者控制端/主设备记录。
- `puppet`: 被遥操作的机器人端数据，即实际执行动作的从端机器人记录。

本地 README 没有提供官方字段解释，因此上述含义是基于数据结构和遥操作命名的判断。

```text
/
├── master/                                        # 遥操作主端数据
│   ├── arm_left_position_align/                   # 左臂关节位置，已和 trajectory 时间步对齐
│   │   ├── data                                   # shape=(213, 8), dtype=float32；每步 8 维，通常可理解为 7 个关节 + 1 个夹爪/末端标量
│   │   ├── timestamp                              # shape=(213,), dtype=int64；该流每步时间戳
│   │   └── is_intervene                           # shape=(213,), dtype=bool；该流每步是否人工介入
│   ├── arm_right_position_align/                  # 右臂关节位置，字段结构同上
│   │   ├── data                                   # shape=(213, 8), dtype=float32
│   │   ├── timestamp                              # shape=(213,), dtype=int64
│   │   └── is_intervene                           # shape=(213,), dtype=bool
│   ├── end_effector_left_pose_align/              # 左末端位姿，已和 trajectory 时间步对齐
│   │   ├── data                                   # shape=(213, 7), dtype=float64；通常为 xyz + quaternion
│   │   ├── timestamp                              # shape=(213,), dtype=int64
│   │   └── is_intervene                           # shape=(213,), dtype=bool
│   ├── end_effector_right_pose_align/             # 右末端位姿，字段结构同上
│   │   ├── data                                   # shape=(213, 7), dtype=float64
│   │   ├── timestamp                              # shape=(213,), dtype=int64
│   │   └── is_intervene                           # shape=(213,), dtype=bool
│   ├── end_effector_left_position_align/          # 左末端/夹爪位置标量
│   │   ├── data                                   # shape=(213, 1), dtype=float32
│   │   ├── timestamp                              # shape=(213,), dtype=int64
│   │   └── is_intervene                           # shape=(213,), dtype=bool
│   └── end_effector_right_position_align/         # 右末端/夹爪位置标量
│       ├── data                                   # shape=(213, 1), dtype=float32
│       ├── timestamp                              # shape=(213,), dtype=int64
│       └── is_intervene                           # shape=(213,), dtype=bool
└── puppet/                                        # 被控制的机器人端数据，字段与 master 相同
    ├── arm_left_position_align/                   # 左臂关节位置，已和 trajectory 时间步对齐
    │   ├── data                                   # shape=(213, 8), dtype=float32
    │   ├── timestamp                              # shape=(213,), dtype=int64
    │   └── is_intervene                           # shape=(213,), dtype=bool
    ├── arm_right_position_align/                  # 右臂关节位置，已和 trajectory 时间步对齐
    │   ├── data                                   # shape=(213, 8), dtype=float32
    │   ├── timestamp                              # shape=(213,), dtype=int64
    │   └── is_intervene                           # shape=(213,), dtype=bool
    ├── end_effector_left_pose_align/              # 左末端位姿
    │   ├── data                                   # shape=(213, 7), dtype=float64
    │   ├── timestamp                              # shape=(213,), dtype=int64
    │   └── is_intervene                           # shape=(213,), dtype=bool
    ├── end_effector_right_pose_align/             # 右末端位姿
    │   ├── data                                   # shape=(213, 7), dtype=float64
    │   ├── timestamp                              # shape=(213,), dtype=int64
    │   └── is_intervene                           # shape=(213,), dtype=bool
    ├── end_effector_left_position_align/          # 左末端/夹爪位置标量
    │   ├── data                                   # shape=(213, 1), dtype=float32
    │   ├── timestamp                              # shape=(213,), dtype=int64
    │   └── is_intervene                           # shape=(213,), dtype=bool
    └── end_effector_right_position_align/         # 右末端/夹爪位置标量
        ├── data                                   # shape=(213, 1), dtype=float32
        ├── timestamp                              # shape=(213,), dtype=int64
        └── is_intervene                           # shape=(213,), dtype=bool
```

## 一个 sample 的逻辑组成

第 `i` 个 sample 由所有时序 dataset 的第 `i` 项组合而成：

```text
sample[i]
├── camera_observations
│   ├── timestamp                                  # camera_observations/timestamp[i]
│   ├── is_intervene                               # camera_observations/is_intervene[i]
│   ├── color_images
│   │   ├── camera_front                           # color_images/camera_front[i]
│   │   ├── camera_left                            # color_images/camera_left[i]
│   │   ├── camera_right                           # color_images/camera_right[i]
│   │   ├── camera_top                             # color_images/camera_top[i]
│   │   ├── camera_wrist_left                      # color_images/camera_wrist_left[i]
│   │   └── camera_wrist_right                     # color_images/camera_wrist_right[i]
│   └── depth_images
│       ├── camera_front                           # depth_images/camera_front[i]
│       ├── camera_left                            # depth_images/camera_left[i]
│       ├── camera_right                           # depth_images/camera_right[i]
│       ├── camera_top                             # depth_images/camera_top[i]
│       ├── camera_wrist_left                      # depth_images/camera_wrist_left[i]
│       └── camera_wrist_right                     # depth_images/camera_wrist_right[i]
├── master
│   ├── arm_left_position_align                    # data[i], timestamp[i], is_intervene[i]
│   ├── arm_right_position_align                   # data[i], timestamp[i], is_intervene[i]
│   ├── end_effector_left_pose_align               # data[i], timestamp[i], is_intervene[i]
│   ├── end_effector_right_pose_align              # data[i], timestamp[i], is_intervene[i]
│   ├── end_effector_left_position_align           # data[i], timestamp[i], is_intervene[i]
│   └── end_effector_right_position_align          # data[i], timestamp[i], is_intervene[i]
└── puppet
    ├── arm_left_position_align                    # data[i], timestamp[i], is_intervene[i]
    ├── arm_right_position_align                   # data[i], timestamp[i], is_intervene[i]
    ├── end_effector_left_pose_align               # data[i], timestamp[i], is_intervene[i]
    ├── end_effector_right_pose_align              # data[i], timestamp[i], is_intervene[i]
    ├── end_effector_left_position_align           # data[i], timestamp[i], is_intervene[i]
    └── end_effector_right_position_align          # data[i], timestamp[i], is_intervene[i]
```

## sample[0] 示例

### 相机观测

```text
camera_observations.timestamp: 1744780088
camera_observations.is_intervene: False
```

彩色图像每项是 JPEG 编码后的 `uint8` bytes：

```text
color_images.camera_front:       shape=(212822,), dtype=uint8, bytes=212822
color_images.camera_left:        shape=(111043,), dtype=uint8, bytes=111043
color_images.camera_right:       shape=(102858,), dtype=uint8, bytes=102858
color_images.camera_top:         shape=(234032,), dtype=uint8, bytes=234032
color_images.camera_wrist_left:  shape=(61558,),  dtype=uint8, bytes=61558
color_images.camera_wrist_right: shape=(62596,),  dtype=uint8, bytes=62596
```

深度图像每项是 PNG 编码后的 `uint8` bytes：

```text
depth_images.camera_front:       shape=(295184,), dtype=uint8, bytes=295184
depth_images.camera_left:        shape=(145176,), dtype=uint8, bytes=145176
depth_images.camera_right:       shape=(141645,), dtype=uint8, bytes=141645
depth_images.camera_top:         shape=(342240,), dtype=uint8, bytes=342240
depth_images.camera_wrist_left:  shape=(57975,),  dtype=uint8, bytes=57975
depth_images.camera_wrist_right: shape=(50291,),  dtype=uint8, bytes=50291
```

### master 示例

```text
master.arm_left_position_align:
  timestamp: 1744780088
  is_intervene: False
  data: [0.06749515235424042, 0.10124272853136063, -0.07669888436794281, -1.495631217956543, 0.00920388475060463, 1.5769321918487549, 0.003067961661145091, 0.0]

master.arm_right_position_align:
  timestamp: 1744780088
  is_intervene: False
  data: [0.13962292671203613, 0.09664078801870346, -0.016873788088560104, -1.7395343780517578, -0.04766078293323517, 1.8254371881484985, 0.3097412586212158, 0.0]

master.end_effector_left_pose_align:
  timestamp: 1744780088
  is_intervene: False
  data: [0.5888612866401672, -0.0015317276120185852, 0.5987640619277954, 0.9999181998349661, -0.00439422778033088, -0.010553368686459504, 0.0057367944344282286]

master.end_effector_right_pose_align:
  timestamp: 1744780088
  is_intervene: False
  data: [0.5843474864959717, 0.06860753148794174, 0.5052845478057861, 0.99585636152112, -0.08816074842668747, 0.0006584613411947998, 0.022301481641926902]

master.end_effector_left_position_align:
  timestamp: 1744780088
  is_intervene: False
  data: [0.0]

master.end_effector_right_position_align:
  timestamp: 1744780088
  is_intervene: False
  data: [0.0]
```

### puppet 示例

```text
puppet.arm_left_position_align:
  timestamp: 1744780088
  is_intervene: False
  data: [0.06963855028152466, 0.10993748158216476, -0.07587612420320511, -1.4948431253433228, -0.0031892885453999043, 1.5833749771118164, 0.0029866828117519617, 0.0]

puppet.arm_right_position_align:
  timestamp: 1744780088
  is_intervene: False
  data: [0.13749276101589203, 0.08633773028850555, -0.015123759396374226, -1.7669011354446411, -0.04422019422054291, 1.8448972702026367, 0.31119561195373535, 0.0]

puppet.end_effector_left_pose_align:
  timestamp: 1744780088
  is_intervene: False
  data: [0.5888612866401672, -0.0015317276120185852, 0.5987640619277954, 0.9999181998349661, -0.00439422778033088, -0.010553368686459504, 0.0057367944344282286]

puppet.end_effector_right_pose_align:
  timestamp: 1744780088
  is_intervene: False
  data: [0.5843474864959717, 0.06860753148794174, 0.5052845478057861, 0.99585636152112, -0.08816074842668747, 0.0006584613411947998, 0.022301481641926902]

puppet.end_effector_left_position_align:
  timestamp: 1744780088
  is_intervene: False
  data: [0.0]

puppet.end_effector_right_position_align:
  timestamp: 1744780088
  is_intervene: False
  data: [0.0]
```
