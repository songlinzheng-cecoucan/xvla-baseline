## ADDED Requirements

### Requirement: XVLA 相关 Xsens 子集选择

baseline training dataset MUST 优先选择与 XVLA 开放任务存在明确技能近邻的 RoboMIND TienKung Xsens task，而不是盲目下载整个 Xsens 子目录。

#### Scenario: 记录任务选择依据

- **GIVEN** 一个候选 Xsens compressed task 名称
- **WHEN** 该 task 被纳入 baseline 下载或训练计划
- **THEN** 文档或 manifest 必须记录其来源目录、task 名称和与 XVLA 任务的相关类别
- **AND** 相关类别必须至少区分为齿轮/工业零件放置、按钮/开关操作、小物体放入托盘/盘子、插入/拔出/精密装配、推/压/接触操作之一
- **AND** 不得仅因为 task 属于 `h5_tienkung_xsens_1rgb` 就默认纳入训练集

#### Scenario: 优先选择高相关任务

- **GIVEN** XVLA 任务包含齿轮放托盘、工业开关/按钮放置、双手小物体放置和门/侧门开关
- **WHEN** 构建下一版 Xsens baseline dataset
- **THEN** 应优先选择 `gear_place`、`place_button`、`place_button_then_press`、`pick_shelf_insert_machine_press_switch_place_plate` 和 `pick_pipe_place_plate_twice`
- **AND** 可以将 `battery_insertion_with_pullout`、`plug_insertion`、`plug_insertion_v2`、`pick_cup_pour_cup_place`、`push_cup_pick_cup_insert_cup` 作为中高相关补充数据
- **AND** 必须标注这些任务是技能近邻，不得标注为 XVLA 同构数据

### Requirement: 多任务 Xsens Dataset Manifest

多任务 Xsens 转换流程 MUST 输出 task distribution manifest，用于记录每个任务的数据量和来源。

#### Scenario: 写出 task distribution manifest

- **GIVEN** 多个 Xsens task 的 HDF5 trajectory 被转换为统一 LeRobotDataset
- **WHEN** conversion 完成
- **THEN** 必须写出 task distribution manifest
- **AND** manifest 必须包含每个 task 的 `num_episodes`、`num_frames` 和 source directory
- **AND** manifest 必须包含 total episodes、total frames 和 canonical action dim
- **AND** 不得只记录全局 episode 数而丢失 task 分布

### Requirement: 多任务评估

多任务 baseline evaluation MUST 同时报告 global 指标和 per-task 指标。

#### Scenario: 计算 per-task loss

- **GIVEN** 一个多任务 Xsens LeRobotDataset
- **AND** 一个 episode-level split manifest
- **WHEN** validation/test loss 脚本运行
- **THEN** 必须能按 task 聚合 flow-matching loss
- **AND** 必须输出 global mean loss
- **AND** 必须输出每个 task 的 mean loss、num_samples 和 episode 列表

#### Scenario: 渲染高相关 task 的 open-loop 视频

- **GIVEN** 一个多任务 baseline checkpoint
- **AND** held-out split 中包含高优先级 task
- **WHEN** 生成 open-loop 可视化
- **THEN** 至少应为 `gear_place`、`place_button` 和 `place_button_then_press` 各渲染一个 held-out episode
- **AND** 视频说明必须标注这是 open-loop imitation visualization，不是 XVLA benchmark success rate
