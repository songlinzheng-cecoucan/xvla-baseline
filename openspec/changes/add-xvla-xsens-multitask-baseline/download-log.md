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
