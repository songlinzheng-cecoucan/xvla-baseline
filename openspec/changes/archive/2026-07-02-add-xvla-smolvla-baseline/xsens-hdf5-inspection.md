# TienKung Xsens HDF5 检查记录

已检查样本：

```text
/tmp/robomind_xsens_extract/pick_pipe_place_plate_twice/success_episodes/train/2024-11-06-10-22-02/data/trajectory.hdf5
```

来源 archive shard：

```text
/home/slzheng/datasets/RoboMIND_smoke/tienkung_xsens_pick_pipe/pick_pipe_place_plate_twice.tar.gz.part-aa
```

该 shard 是 partial download，但第一个 `trajectory.hdf5` 可以被抽出，并且能用 `h5py` 正常打开。

## Metadata

```text
attrs:
  compress: True
  sim: False

language_raw:
  pick pipe place plate twice

language_distilbert:
  shape=(1, 1, 768)
  dtype=float16
```

## Groups 和 Datasets

```text
master/end_effector                  shape=(413, 12), dtype=float64
master/joint_position                shape=(413, 14), dtype=float64
observations/depth_images/camera_top shape=(413,), dtype=object
observations/rgb_images/camera_top   shape=(413,), dtype=object
puppet/end_effector                  shape=(413, 12), dtype=float64
puppet/joint_position                shape=(413, 14), dtype=float64
```

## Canonical 26D 映射检查

`master` 和 `puppet` 都可以映射到 canonical 26D 向量：

```text
joint_position[:, 0:7]   -> left_arm
joint_position[:, 7:14]  -> right_arm
end_effector[:, 0:6]     -> left_hand
end_effector[:, 6:12]    -> right_hand
```

实际结果：

```text
master canonical26 shape=(413, 26), dtype=float64, contains_nan=False
puppet canonical26 shape=(413, 26), dtype=float64, contains_nan=False
```

RoboMIND `robomind.yaml` 对 Xsens 声明 `action_arm_key: puppet`，因此第一版 baseline 使用 `puppet` 同时作为 state 和 action 来源。

## 图像检查

```text
observations/rgb_images/camera_top[0]:
  encoded bytes shape=(111747,), dtype=uint8
  decoded image mode=RGB
  decoded size=(640, 480)

observations/depth_images/camera_top[0]:
  encoded bytes shape=(159498,), dtype=uint8
  decoded image mode=I;16
  decoded size=(640, 480)
  decoded dtype=uint16
  decoded min/max=0/8648
```

## 结论

该真实 HDF5 样本满足 OpenSpec smoke-inspection 要求：

- `puppet/joint_position.shape[-1] == 14`
- `puppet/end_effector.shape[-1] == 12`
- `observations/rgb_images/camera_top` 存在并可解码为 RGB
- 可构造无 NaN 的 canonical 26D 向量
- `language_raw` 存在并可作为任务文本

