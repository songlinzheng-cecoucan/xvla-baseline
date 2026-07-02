# Reader 系统设计：RoboMIND TienKung Xsens

## 现有工具评估

公开的 `Open-X-Humanoid/x-humanoid-training-toolchain` 仓库提供了通用 HDF5 到 LeRobot 的转换脚本。当前 `convert_to_lerobot.py` 主要读取：

```text
file["puppet/joint_position"]
file["observations/rgb_images/camera_top"]
```

并把同一个 `puppet_state` 写入：

```text
observation.state
action
```

仓库提供的 `Tien_Kung_Gello_1RGB.json` 配置定义的是 16 维 state/action，对应 TienKung Gello：

```text
左臂 7 + 左手 closure 1 + 右臂 7 + 右手 closure 1
```

这可以作为 LeRobot 转换方式的参考，但不能直接用于 TienKung Xsens。Xsens 存储结构是：

```text
puppet/joint_position: shape=(T, 14)
puppet/end_effector:   shape=(T, 12)
```

因此 XVLA baseline reader 必须组合这两个字段，构造 26 维 canonical vector。

## 目标

reader 必须做到：

- 打开 RoboMIND TienKung Xsens `trajectory.hdf5`。
- 返回数据前先验证 schema、shape、dtype 和核心语义字段。
- 暴露或解码 `camera_top` RGB 帧。
- 构造 canonical 26D state/action。
- 保留 task 和 metadata。
- 为 inspection 工具和 LeRobotDataset converter 提供稳定 API。

reader 不得：

- 静默接受非 Xsens schema。
- 猜测缺失字段。
- 混用 Gello 16D 和 Xsens 26D 格式。
- 在没有显式 mapping table 的情况下做 BrainCo/TianYi 专属 remapping。

## v1 模块布局

第一版采用 package + CLI wrapper 结构：

```text
src/xvla_baseline/data/robomind_xsens_reader.py
src/xvla_baseline/data/convert_xsens_to_lerobot.py
scripts/convert_xsens_to_lerobot.py   # thin CLI wrapper
```

当出现以下情况时再拆成包：

- 单个 reader 文件超过约 300 行。
- 需要支持第二种 embodiment 或第二种 RoboMIND schema。
- 增加 frame iterator、depth 处理、language embedding 处理等独立复杂逻辑。

后续 reader 内部可继续演进为：

```text
src/xvla_baseline/data/robomind_xsens/
  schema.py       # 常量、字段名、期望维度
  reader.py       # HDF5 读取和验证
  sample.py       # dataclass / typed record
  images.py       # 图像 byte 解码 helper
  convert.py      # LeRobotDataset 转换入口
```

## 数据模型

使用一个轻量 trajectory record：

```python
@dataclass
class XsensTrajectory:
    path: Path
    task: str
    num_frames: int
    state: np.ndarray       # (T, 26), float32 或 float64
    action: np.ndarray      # (T, 26), float32 或 float64
    rgb_camera_top: Sequence[np.ndarray] | Sequence[bytes]
    metadata: dict[str, Any]
```

v1 内存模型：

- 一次读取一条 trajectory 到内存。
- batch converter 负责按文件顺序逐条处理。
- reader v1 不实现跨文件 streaming。
- 默认 `decode_images=False`，即保留 encoded image bytes，避免不必要的内存膨胀。

## Task 和 Metadata

`task` 必须来自 HDF5 的 `language_raw[0]`。

如果 `language_raw` 缺失、为空、或无法解码为字符串，reader 必须抛出 schema error。文件夹名可以进入 metadata，但不能作为默认 task fallback。

metadata 至少包含：

```text
path
source
action_mode
language_raw
has_language_distilbert
num_frames
rgb_camera_top_size
sim
compress
image_key
benchmark_image_key_hint
```

其中：

```text
image_key = observations/rgb_images/camera_top
benchmark_image_key_hint = camera_head
```

`benchmark_image_key_hint` 只提醒下游：RoboMIND 的 `camera_top` 与 XVLA benchmark 的 `camera_head` 不是同一个相机语义。reader 不做自动重命名。

## Canonical 26D 构造

输入字段：

```text
joint_position: (T, 14)
end_effector:   (T, 12)
```

输出：

```text
canonical26 = concat(
    joint_position[:, 0:7],    # left_arm
    joint_position[:, 7:14],   # right_arm
    end_effector[:, 0:6],      # left_hand
    end_effector[:, 6:12],     # right_hand
)
```

RoboMIND 文档中的 Xsens hand 6D 顺序：

```text
[小指, 无名指, 中指, 食指, 拇指0弯曲, 拇指1旋转]
```

v1 只支持 `source="puppet"`，因为 RoboMIND `robomind.yaml` 声明：

```text
h5_tienkung_xsens_1rgb:
  qpos_arm_key: puppet
  action_arm_key: puppet
  controls: [end_effector, joint_position]
```

如果后续要支持 `master`，必须新增明确任务和测试，不在 v1 reader 中预先承诺。

## State/Action 语义

v1 使用 absolute target 语义：

```text
observation.state[t] = puppet canonical26[t]
action[t] = puppet canonical26[t]
```

理由：

- XVLA benchmark action payload 要求目标关节位置。
- RoboMIND Xsens schema 声明 `action_arm_key=puppet`。
- smoke baseline 的首要目标是跑通数据、模型和接口，不先引入 delta 或 next-state target。

后续可作为实验扩展比较：

```text
next_state:
  observation.state[t] = canonical26[t]
  action[t] = canonical26[t + 1]

delta:
  observation.state[t] = canonical26[t]
  action[t] = canonical26[t + 1] - canonical26[t]
```

## Validation 规则

reader 必须验证：

```text
language_raw exists
language_raw[0] 可解码为非空字符串
puppet/joint_position exists
puppet/end_effector exists
observations/rgb_images/camera_top exists
joint_position 是数值 dtype
end_effector 是数值 dtype
joint_position.ndim == 2
end_effector.ndim == 2
joint_position.shape[0] > 0
joint_position.shape[1] == 14
end_effector.shape[1] == 12
joint_position.shape[0] == end_effector.shape[0]
camera_top.ndim == 1
camera_top.shape[0] == joint_position.shape[0]
camera_top dtype 是 object 或可索引 encoded bytes 序列
canonical26 不包含 NaN 或 inf
至少第 0 帧 RGB 可用 Pillow/OpenCV 解码
第 0 帧 RGB mode 为 RGB 或可转换为 RGB
```

图像尺寸不要求固定写死为 640x480，但 reader 必须记录首帧尺寸到 metadata。converter 可以选择进一步要求全轨迹尺寸一致。

## Schema Error

Validation 失败必须抛出自定义异常：

```python
class XsensSchemaError(ValueError):
    def __init__(
        self,
        *,
        path: str | Path,
        field: str,
        expected: str,
        actual: str,
        reason: str,
    ) -> None:
        ...
```

错误信息必须包含：

```text
path
field
expected
actual
reason
```

这样 batch conversion 时可以统计“缺 camera_top”“维度不对”“图像无法解码”等失败类型。

## 图像处理

RoboMIND 将 RGB 帧保存为 encoded byte arrays：

```text
observations/rgb_images/camera_top: shape=(T,), dtype=object
```

reader 提供两种模式：

```text
decode_images=False:
  返回 encoded bytes，适合后续转换或视频编码。

decode_images=True:
  返回 RGB arrays，shape=(H, W, 3)，适合 smoke test 和可视化。
```

图像解码必须使用 Pillow 或 OpenCV，不得手写 byte parser。

## Public API

初始 reader API：

```python
def read_xsens_trajectory(
    path: str | Path,
    *,
    decode_images: bool = False,
    dtype: np.dtype = np.float32,
) -> XsensTrajectory:
    ...
```

Inspection helper：

```python
def inspect_xsens_trajectory(path: str | Path) -> dict[str, Any]:
    ...
```

Canonical mapping helper：

```python
def build_canonical26(joint_position: np.ndarray, end_effector: np.ndarray) -> np.ndarray:
    ...
```

注意：v1 API 不暴露 `source` 参数。`source="master"` 支持需要单独设计、验证和测试。

## Smoke Test

当前可用真实样本：

```text
/tmp/robomind_xsens_extract/pick_pipe_place_plate_twice/success_episodes/train/2024-11-06-10-22-02/data/trajectory.hdf5
```

该样本来自 partial shard：

```text
/home/slzheng/datasets/RoboMIND_smoke/tienkung_xsens_pick_pipe/pick_pipe_place_plate_twice.tar.gz.part-aa
```

测试路径不得写死在库代码中。测试脚本应优先读取环境变量：

```text
ROBOMIND_XSENS_SAMPLE=/path/to/trajectory.hdf5
```

最小检查：

```text
read_xsens_trajectory(...).state.shape == (413, 26)
read_xsens_trajectory(...).action.shape == (413, 26)
task == "pick pipe place plate twice"
metadata["rgb_camera_top_size"] == (640, 480)
第 0 帧 RGB 可解码
错误 schema 会抛出 XsensSchemaError
```

## LeRobot Converter 依赖关系

converter 必须依赖 reader，不得重新实现 schema 逻辑。

流程：

```text
HDF5 path
-> read_xsens_trajectory()
-> frame loop
-> LeRobotDataset.add_frame({
     "task": trajectory.task,
     "observation.state": trajectory.state[t],
     "action": trajectory.action[t],
     "observation.images.camera_top": rgb[t],
   })
```

这样 schema validation 可以被 inspection、dataset conversion 和后续 policy debugging 复用。
