# 提案：构建 XVLA 相关 Xsens 多任务 Baseline 数据集

## 背景

`add-xvla-smolvla-baseline` 已经完成 smoke / 单任务 baseline 链路：RoboMIND TienKung Xsens reader、LeRobotDataset 转换、SmolVLA LoRA 训练、validation loss 和 open-loop inspection 都已跑通。

下一步不应继续只在 `pick_pipe_place_plate_twice` 上加训练步数，而应选择更贴近 XVLA 公开任务的 Xsens task，形成多任务数据集：

- 齿轮/工业零件放置。
- 工业开关/红色按钮放置和按压。
- 小物体放入托盘/盘子。
- 插入/拔出/精密装配。
- 接触、推、压和简单开合操作。

## 范围

本 change 覆盖：

- 整理 `benchmark1_0_compressed` 和 `benchmark1_1_compressed` 的 TienKung Xsens task。
- 按 XVLA 公开任务相关性对 task 分类和排序。
- 下载高相关 task 的 compressed shards，并解压为 HDF5 trajectory。
- 复用既有 Xsens reader/converter，构建统一多任务 LeRobotDataset。
- 生成 episode-level train/val/test split 和 task distribution manifest。
- 在多任务 dataset 上训练下一版 SmolVLA LoRA baseline。
- 复用 validation loss、open-loop inspection 和视频渲染工具评估 checkpoint。

## 不在范围内

- 下载完整 RoboMIND 或完整 `h5_tienkung_xsens_1rgb/**`。
- 保证 Xsens task 与 XVLA 任务同物体、同场景、同成功条件。
- 实现 ZMQ policy service 或 Isaac Sim 闭环 rollout。
- 修改 SmolVLA 算法、loss 或 LeRobot 训练内核。
- 解决 TianYi2.0/BrainCo2 与 TienKung/Inspire hand 的精确动作重映射。

## 风险

- Hugging Face 数据集是 gated dataset，目录完整性需要登录态确认。
- `benchmark1_0_compressed` 网页存在 `Load more files`，页面预览不是完整清单。
- 每个 compressed task 可能包含多个大分片，下载和解压会占用几十 GB 到上百 GB。
- 任务名只提供语义近邻，不等价于 XVLA 同构示教数据。
- 多任务训练可能降低当前单任务 loss，但更接近比赛任务分布；评估时必须同时看 per-task loss 和 held-out open-loop 视频。
