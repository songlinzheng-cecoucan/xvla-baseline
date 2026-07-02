# 仓库协作约定

本仓库使用 OpenSpec 做 spec-first 任务管理。

- 实现 active change 前，先阅读对应的 `proposal.md`、`design.md`、`tasks.md` 和相关 spec delta。
- `openspec/specs/` 是长期稳定契约层。
- `openspec/changes/` 是正在规划或实施的变更队列。
- 不要把大型数据集、checkpoint、生成视频或原始 HDF5 文件提交进 git。
- 本地数据集放在 `/home/slzheng/datasets`。
- 模型权重优先使用 Hugging Face cache。
- Isaac Sim 运行环境和训练环境保持分离：`xmimic` 用于仿真，`lerobot312` 用于训练和数据转换。

