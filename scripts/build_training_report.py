#!/usr/bin/env python3
"""Build a compact training/evaluation report for the balanced232 SmolVLA run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageDraw, ImageFont


RUN_NAME = "balanced232_050000"
DEFAULT_LOG = Path(
    "/home/slzheng/datasets/xvla/runs/logs/"
    "smolvla_lora_xvla_multitask_v4_balanced232_50000_bs2_nw2.log"
)
DEFAULT_VAL_JSON = Path("/home/slzheng/datasets/xvla/eval_reports/balanced232_050000/val_per_task_loss.json")
DEFAULT_TEST_JSON = Path("/home/slzheng/datasets/xvla/eval_reports/balanced232_050000/test_per_task_loss.json")
DEFAULT_INSPECTION_SUMMARY = Path(
    "/home/slzheng/datasets/xvla/open_loop_inspections/balanced232_050000_test_200/summary.json"
)
DEFAULT_VIDEO_DIR = Path("/home/slzheng/datasets/xvla/open_loop_videos/balanced232_050000")
DEFAULT_OUTPUT_DIR = Path("docs/experiments/balanced232_050000")

TRAIN_RE = re.compile(
    r"Training:\s+.*?\|\s*(?P<progress_step>\d+)/50000.*?"
    r"INFO\s+(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"step:(?P<step>\S+)\s+smpl:(?P<samples>\S+)\s+ep:(?P<episode>\d+)\s+"
    r"epch:(?P<epoch>[0-9.]+)\s+loss:(?P<loss>[0-9.eE+-]+)\s+"
    r"grdn:(?P<grad_norm>[0-9.eE+-]+)\s+lr:(?P<lr>[0-9.eE+-]+)\s+"
    r"updt_s:(?P<update_s>[0-9.eE+-]+)\s+data_s:(?P<data_s>[0-9.eE+-]+)"
)


@dataclass(frozen=True)
class TrainPoint:
    timestamp: datetime
    step: int
    samples: str
    episode: int
    epoch: float
    loss: float
    grad_norm: float
    lr: float
    update_s: float
    data_s: float


def parse_train_log(path: Path) -> list[TrainPoint]:
    points: list[TrainPoint] = []
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    for line in text.splitlines():
        match = TRAIN_RE.search(line)
        if not match:
            continue
        groups = match.groupdict()
        points.append(
            TrainPoint(
                timestamp=datetime.strptime(groups["timestamp"], "%Y-%m-%d %H:%M:%S"),
                step=int(groups["progress_step"]),
                samples=groups["samples"],
                episode=int(groups["episode"]),
                epoch=float(groups["epoch"]),
                loss=float(groups["loss"]),
                grad_norm=float(groups["grad_norm"]),
                lr=float(groups["lr"]),
                update_s=float(groups["update_s"]),
                data_s=float(groups["data_s"]),
            )
        )
    if not points:
        raise ValueError(f"No training points parsed from {path}")
    return points


def moving_average(values: list[float], window: int) -> list[float]:
    result: list[float] = []
    total = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        result.append(total / len(queue))
    return result


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_train_csv(path: Path, points: list[TrainPoint]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "step",
                "samples",
                "episode",
                "epoch",
                "loss",
                "grad_norm",
                "lr",
                "update_s",
                "data_s",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "timestamp": point.timestamp.isoformat(sep=" "),
                    "step": point.step,
                    "samples": point.samples,
                    "episode": point.episode,
                    "epoch": point.epoch,
                    "loss": point.loss,
                    "grad_norm": point.grad_norm,
                    "lr": point.lr,
                    "update_s": point.update_s,
                    "data_s": point.data_s,
                }
            )


def nearest_checkpoints(points: list[TrainPoint], every: int = 5000) -> list[TrainPoint]:
    checkpoints = list(range(every, 50001, every))
    selected: list[TrainPoint] = []
    for checkpoint in checkpoints:
        selected.append(min(points, key=lambda item: abs(item.step - checkpoint)))
    return selected


def write_checkpoint_csv(path: Path, points: list[TrainPoint]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["checkpoint_step", "logged_step", "timestamp", "loss", "grad_norm", "lr", "data_s", "update_s"],
            lineterminator="\n",
        )
        writer.writeheader()
        for checkpoint, point in zip(range(5000, 50001, 5000), nearest_checkpoints(points), strict=True):
            writer.writerow(
                {
                    "checkpoint_step": checkpoint,
                    "logged_step": point.step,
                    "timestamp": point.timestamp.isoformat(sep=" "),
                    "loss": point.loss,
                    "grad_norm": point.grad_norm,
                    "lr": point.lr,
                    "data_s": point.data_s,
                    "update_s": point.update_s,
                }
            )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_eval_csv(path: Path, val: dict[str, Any], test: dict[str, Any]) -> None:
    ensure_parent(path)
    rows = [
        {
            "scope": "global",
            "val_loss": val["mean_loss"],
            "val_samples": val["num_samples"],
            "test_loss": test["mean_loss"],
            "test_samples": test["num_samples"],
        }
    ]
    for task in sorted(val["per_task"]):
        rows.append(
            {
                "scope": task,
                "val_loss": val["per_task"][task]["mean_loss"],
                "val_samples": val["per_task"][task]["num_samples"],
                "test_loss": test["per_task"][task]["mean_loss"],
                "test_samples": test["per_task"][task]["num_samples"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scope", "val_loss", "val_samples", "test_loss", "test_samples"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def get_font(size: int = 16) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_axes(
    draw: ImageDraw.ImageDraw,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    title: str,
    x_label: str,
    y_label: str,
    y_min: float,
    y_max: float,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    draw.text((x0, 20), title, fill="#111111", font=font)
    draw.line([(x0, y1), (x1, y1), (x1, y0)], fill="#333333", width=2)
    draw.line([(x0, y0), (x0, y1)], fill="#333333", width=2)
    for tick in range(6):
        ratio = tick / 5
        y = y1 - int((y1 - y0) * ratio)
        value = y_min + (y_max - y_min) * ratio
        draw.line([(x0 - 5, y), (x1, y)], fill="#e6e6e6")
        draw.text((8, y - 8), f"{value:.3g}", fill="#444444", font=small_font)
    draw.text(((x0 + x1) // 2 - 40, y1 + 38), x_label, fill="#333333", font=small_font)
    draw.text((8, y0 - 28), y_label, fill="#333333", font=small_font)


def scale_point(
    x: float,
    y: float,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[int, int]:
    x_ratio = 0.0 if x_max == x_min else (x - x_min) / (x_max - x_min)
    y_ratio = 0.0 if y_max == y_min else (y - y_min) / (y_max - y_min)
    px = x0 + int((x1 - x0) * x_ratio)
    py = y1 - int((y1 - y0) * y_ratio)
    return px, py


def draw_line_chart(
    path: Path,
    *,
    title: str,
    series: dict[str, tuple[list[float], list[float], str]],
    x_label: str = "step",
    y_label: str = "value",
    width: int = 1200,
    height: int = 620,
) -> None:
    ensure_parent(path)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = get_font(22)
    small_font = get_font(14)
    x0, y0, x1, y1 = 82, 70, width - 35, height - 82

    all_x = [value for values, _, _ in series.values() for value in values]
    all_y = [value for _, values, _ in series.values() for value in values if math.isfinite(value)]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    pad = (y_max - y_min) * 0.08 or 1.0
    y_min -= pad
    y_max += pad
    if min(all_y) >= 0:
        y_min = max(0.0, y_min)

    draw_axes(
        draw,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_min=y_min,
        y_max=y_max,
        font=font,
        small_font=small_font,
    )
    legend_x = x0 + 8
    for idx, (name, (xs, ys, color)) in enumerate(series.items()):
        points = [
            scale_point(x, y, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, x0=x0, y0=y0, x1=x1, y1=y1)
            for x, y in zip(xs, ys, strict=True)
            if math.isfinite(y)
        ]
        if len(points) >= 2:
            draw.line(points, fill=color, width=2 if idx else 1)
        ly = y1 + 14 + idx * 18
        draw.line([(legend_x, ly + 7), (legend_x + 28, ly + 7)], fill=color, width=3)
        draw.text((legend_x + 36, ly), name, fill="#222222", font=small_font)
    image.save(path)


def draw_grouped_bar_chart(
    path: Path,
    *,
    title: str,
    labels: list[str],
    values_a: list[float],
    values_b: list[float],
    legend_a: str,
    legend_b: str,
    width: int = 1200,
    height: int = 640,
) -> None:
    ensure_parent(path)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = get_font(22)
    small_font = get_font(13)
    x0, y0, x1, y1 = 82, 70, width - 35, height - 140
    max_value = max(values_a + values_b + [1e-9])
    draw_axes(
        draw,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        title=title,
        x_label="task",
        y_label="loss",
        y_min=0.0,
        y_max=max_value * 1.2,
        font=font,
        small_font=small_font,
    )
    group_w = (x1 - x0) / len(labels)
    bar_w = min(70, int(group_w * 0.28))
    for idx, label in enumerate(labels):
        center = x0 + group_w * (idx + 0.5)
        for offset, value, color in ((-bar_w * 0.6, values_a[idx], "#4c78a8"), (bar_w * 0.6, values_b[idx], "#f58518")):
            bx0 = int(center + offset - bar_w / 2)
            bx1 = int(center + offset + bar_w / 2)
            by = y1 - int((y1 - y0) * value / (max_value * 1.2))
            draw.rectangle([(bx0, by), (bx1, y1)], fill=color)
            draw.text((bx0, by - 18), f"{value:.3f}", fill="#222222", font=small_font)
        words = label.replace("_", "\n")
        draw.multiline_text((int(center - group_w * 0.38), y1 + 12), words, fill="#333333", font=small_font, spacing=0)
    draw.rectangle([(x0 + 8, y0 + 12), (x0 + 26, y0 + 30)], fill="#4c78a8")
    draw.text((x0 + 34, y0 + 10), legend_a, fill="#222222", font=small_font)
    draw.rectangle([(x0 + 130, y0 + 12), (x0 + 148, y0 + 30)], fill="#f58518")
    draw.text((x0 + 156, y0 + 10), legend_b, fill="#222222", font=small_font)
    image.save(path)


def draw_bar_chart(
    path: Path,
    *,
    title: str,
    labels: list[str],
    values: list[float],
    y_label: str,
    color: str = "#54a24b",
    width: int = 1000,
    height: int = 580,
) -> None:
    ensure_parent(path)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = get_font(22)
    small_font = get_font(13)
    x0, y0, x1, y1 = 82, 70, width - 35, height - 125
    max_value = max(values + [1e-9])
    draw_axes(
        draw,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        title=title,
        x_label="group/task",
        y_label=y_label,
        y_min=0.0,
        y_max=max_value * 1.2,
        font=font,
        small_font=small_font,
    )
    group_w = (x1 - x0) / len(labels)
    bar_w = min(90, int(group_w * 0.55))
    for idx, (label, value) in enumerate(zip(labels, values, strict=True)):
        center = x0 + group_w * (idx + 0.5)
        bx0 = int(center - bar_w / 2)
        bx1 = int(center + bar_w / 2)
        by = y1 - int((y1 - y0) * value / (max_value * 1.2))
        draw.rectangle([(bx0, by), (bx1, y1)], fill=color)
        draw.text((bx0, by - 18), f"{value:.3f}", fill="#222222", font=small_font)
        draw.multiline_text((int(center - group_w * 0.32), y1 + 12), label.replace("_", "\n"), fill="#333333", font=small_font)
    image.save(path)


def task_short_name(task: str) -> str:
    return {
        "gear_place": "gear_place",
        "pick_pipe_place_plate_twice": "pick_pipe",
        "pick_shelf_insert_machine_press_switch_place_plate": "pick_shelf_switch",
        "place_button": "place_button",
    }.get(task, task)


def load_video_summaries(video_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(video_dir.glob("*.json")):
        summaries.append(load_json(path))
    return summaries


def pct_change(first: float, last: float) -> float:
    return 100.0 * (last - first) / first


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_markdown(
    path: Path,
    *,
    points: list[TrainPoint],
    checkpoint_points: list[TrainPoint],
    val: dict[str, Any],
    test: dict[str, Any],
    inspection: dict[str, Any],
    video_summaries: list[dict[str, Any]],
    assets_dir_name: str,
    data_dir_name: str,
    source_paths: dict[str, Path],
) -> None:
    ensure_parent(path)
    first, last = points[0], points[-1]
    last_20 = points[-20:]
    duration = last.timestamp - first.timestamp
    report = []
    report.append("# balanced232_050000 训练与评估报告")
    report.append("")
    report.append("> 本报告整理 SmolVLA LoRA `balanced232_050000` 的训练日志、held-out val/test loss、open-loop action 指标和视频产物。")
    report.append("> 所有指标都是 offline imitation / open-loop sanity check，不代表 XVLA Isaac Sim 闭环成功率。")
    report.append("")
    report.append("## 1. 运行配置")
    report.append("")
    report.append("| 项目 | 值 |")
    report.append("| --- | --- |")
    report.append("| dataset | `local/robomind_xsens_xvla_multitask_v4_balanced_232` |")
    report.append("| episodes / frames | `232 / 129,186` |")
    report.append("| train / val / test episodes | `184 / 24 / 24` |")
    report.append("| per-task train / val / test | `46 / 6 / 6` |")
    report.append("| base policy | `lerobot/smolvla_base` |")
    report.append("| LoRA rank | `16` |")
    report.append("| steps | `50,000` |")
    report.append("| batch size / num_workers | `2 / 2` |")
    report.append("| final checkpoint | `/home/slzheng/datasets/xvla/runs/smolvla_lora_xvla_multitask_v4_balanced232_50000_bs2_nw2/checkpoints/050000/pretrained_model` |")
    report.append("")
    report.append("## 2. 训练过程")
    report.append("")
    report.append("| 指标 | 值 |")
    report.append("| --- | ---: |")
    report.append(f"| 解析到的 log 点数 | {len(points)} |")
    report.append(f"| 首个记录 step / loss | {first.step} / {format_float(first.loss)} |")
    report.append(f"| 最后记录 step / loss | {last.step} / {format_float(last.loss)} |")
    report.append(f"| loss 相对变化 | {pct_change(first.loss, last.loss):.1f}% |")
    report.append(f"| 最后 20 个 log 点平均 loss | {format_float(mean(point.loss for point in last_20))} |")
    report.append(f"| 最后 20 个 log 点平均 grad norm | {format_float(mean(point.grad_norm for point in last_20))} |")
    report.append(f"| 最后 20 个 log 点平均 data_s / update_s | {format_float(mean(point.data_s for point in last_20))} / {format_float(mean(point.update_s for point in last_20))} |")
    report.append(f"| log 时间跨度 | `{duration}` |")
    report.append("")
    report.append("![训练 loss 曲线](assets/train_loss_curve.png)")
    report.append("")
    report.append("![学习率和梯度范数](assets/train_lr_grad_curve.png)")
    report.append("")
    report.append("![数据加载和更新耗时](assets/train_time_curve.png)")
    report.append("")
    report.append("### Checkpoint 附近训练 loss")
    report.append("")
    report.append("| checkpoint step | logged step | loss | grad norm | lr | data_s |")
    report.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for checkpoint, point in zip(range(5000, 50001, 5000), checkpoint_points, strict=True):
        report.append(
            f"| {checkpoint} | {point.step} | {format_float(point.loss)} | "
            f"{format_float(point.grad_norm)} | {point.lr:.2e} | {format_float(point.data_s)} |"
        )
    report.append("")
    report.append("## 3. Held-out Val/Test Flow-Matching Loss")
    report.append("")
    report.append("![val/test per-task loss](assets/val_test_per_task_loss.png)")
    report.append("")
    report.append("| scope | val loss | val samples | test loss | test samples |")
    report.append("| --- | ---: | ---: | ---: | ---: |")
    report.append(
        f"| global | {format_float(val['mean_loss'])} | {val['num_samples']} | "
        f"{format_float(test['mean_loss'])} | {test['num_samples']} |"
    )
    for task in sorted(val["per_task"]):
        v_item = val["per_task"][task]
        t_item = test["per_task"][task]
        report.append(
            f"| `{task}` | {format_float(v_item['mean_loss'])} | {v_item['num_samples']} | "
            f"{format_float(t_item['mean_loss'])} | {t_item['num_samples']} |"
        )
    report.append("")
    report.append("观察：")
    report.append("")
    report.append("- `val` 和 `test` global loss 接近，当前没有明显只记住 validation split 的信号。")
    report.append("- `pick_pipe_place_plate_twice` 和 `pick_shelf_insert_machine_press_switch_place_plate` 的 held-out loss 较低。")
    report.append("- `gear_place` 的 test loss 最高，应优先结合视频检查相机视角、动作幅度和任务内部多样性。")
    report.append("- `place_button` 的 val/test 都偏高，后续可以增加该类数据或单独看右臂/手部误差。")
    report.append("")
    report.append("## 4. Open-Loop Action 指标")
    report.append("")
    report.append("### 200-sample test inspection")
    report.append("")
    report.append("![open-loop group MAE](assets/open_loop_group_mae.png)")
    report.append("")
    report.append("| group | MAE | MSE |")
    report.append("| --- | ---: | ---: |")
    report.append(f"| full | {format_float(inspection['full_mae'])} | {format_float(inspection['full_mse'])} |")
    for group_name, item in inspection["groups"].items():
        report.append(f"| `{group_name}` | {format_float(item['mae'])} | {format_float(item['mse'])} |")
    report.append("")
    report.append("右臂 `right_arm` MAE 最高，是当前 open-loop action error 的主要来源。")
    report.append("")
    report.append("### Held-out video summaries")
    report.append("")
    report.append("![open-loop video full MAE](assets/open_loop_video_mae.png)")
    report.append("")
    report.append("| episode | video | frames | full MAE | left arm | right arm | left hand | right hand |")
    report.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in video_summaries:
        output_path = Path(item["output_path"])
        report.append(
            f"| {item['episode']} | `{output_path.name}` | {item['num_video_frames']} | "
            f"{format_float(item['full_mae'])} | "
            f"{format_float(item['groups']['left_arm']['mae'])} | "
            f"{format_float(item['groups']['right_arm']['mae'])} | "
            f"{format_float(item['groups']['left_hand']['mae'])} | "
            f"{format_float(item['groups']['right_hand']['mae'])} |"
        )
    report.append("")
    report.append("## 5. 文件索引")
    report.append("")
    report.append("| 类型 | 路径 |")
    report.append("| --- | --- |")
    for name, source_path in source_paths.items():
        report.append(f"| {name} | `{source_path}` |")
    report.append(f"| generated CSV | `{data_dir_name}/training_metrics.csv` |")
    report.append(f"| checkpoint CSV | `{data_dir_name}/checkpoint_train_loss.csv` |")
    report.append(f"| eval CSV | `{data_dir_name}/eval_loss_summary.csv` |")
    report.append("")
    report.append("## 6. 结论")
    report.append("")
    report.append("当前 50k baseline 已经明显学到 offline imitation 分布，训练 loss 从 4.176 降到 0.219，held-out global loss 在 `0.28` 左右。")
    report.append("下一步若继续提高比赛相关性，应优先处理两件事：一是查看 `gear_place` / `place_button` 视频定位误差来源；二是等待或实现 ZMQ/Isaac Sim 闭环后，用真实 benchmark success rate 评估。")
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def build_report(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser()
    assets_dir = output_dir / "assets"
    data_dir = output_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_points = parse_train_log(Path(args.train_log).expanduser())
    checkpoint_points = nearest_checkpoints(train_points)
    val = load_json(Path(args.val_json).expanduser())
    test = load_json(Path(args.test_json).expanduser())
    inspection = load_json(Path(args.inspection_summary).expanduser())
    video_summaries = load_video_summaries(Path(args.video_dir).expanduser())

    write_train_csv(data_dir / "training_metrics.csv", train_points)
    write_checkpoint_csv(data_dir / "checkpoint_train_loss.csv", train_points)
    write_eval_csv(data_dir / "eval_loss_summary.csv", val, test)

    steps = [point.step for point in train_points]
    losses = [point.loss for point in train_points]
    smooth_loss = moving_average(losses, window=20)
    draw_line_chart(
        assets_dir / "train_loss_curve.png",
        title="Training loss, 50k steps",
        y_label="flow-matching loss",
        series={
            "raw loss": (steps, losses, "#b8b8b8"),
            "moving average, window=20": (steps, smooth_loss, "#4c78a8"),
        },
    )
    draw_line_chart(
        assets_dir / "train_lr_grad_curve.png",
        title="Learning rate and grad norm",
        y_label="scaled value",
        series={
            "grad norm": (steps, [point.grad_norm for point in train_points], "#e45756"),
            "lr x 10000": (steps, [point.lr * 10000.0 for point in train_points], "#4c78a8"),
        },
    )
    draw_line_chart(
        assets_dir / "train_time_curve.png",
        title="Data loading and update time",
        y_label="seconds",
        series={
            "data_s": (steps, [point.data_s for point in train_points], "#54a24b"),
            "update_s": (steps, [point.update_s for point in train_points], "#f58518"),
        },
    )

    task_names = sorted(val["per_task"])
    draw_grouped_bar_chart(
        assets_dir / "val_test_per_task_loss.png",
        title="Held-out val/test loss by task",
        labels=[task_short_name(task) for task in task_names],
        values_a=[float(val["per_task"][task]["mean_loss"]) for task in task_names],
        values_b=[float(test["per_task"][task]["mean_loss"]) for task in task_names],
        legend_a="validation",
        legend_b="test",
    )
    group_names = ["left_arm", "right_arm", "left_hand", "right_hand"]
    draw_bar_chart(
        assets_dir / "open_loop_group_mae.png",
        title="Open-loop group MAE, test 200 samples",
        labels=group_names,
        values=[float(inspection["groups"][name]["mae"]) for name in group_names],
        y_label="MAE",
        color="#54a24b",
    )
    draw_bar_chart(
        assets_dir / "open_loop_video_mae.png",
        title="Open-loop video full MAE by held-out episode",
        labels=[Path(item["output_path"]).stem.replace("_ep", "\nep") for item in video_summaries],
        values=[float(item["full_mae"]) for item in video_summaries],
        y_label="full MAE",
        color="#4c78a8",
        width=1200,
    )

    report_path = output_dir / "report.md"
    write_markdown(
        report_path,
        points=train_points,
        checkpoint_points=checkpoint_points,
        val=val,
        test=test,
        inspection=inspection,
        video_summaries=video_summaries,
        assets_dir_name="assets",
        data_dir_name="data",
        source_paths={
            "training log": Path(args.train_log).expanduser(),
            "validation loss JSON": Path(args.val_json).expanduser(),
            "test loss JSON": Path(args.test_json).expanduser(),
            "open-loop inspection summary": Path(args.inspection_summary).expanduser(),
            "open-loop video dir": Path(args.video_dir).expanduser(),
        },
    )
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-log", default=str(DEFAULT_LOG))
    parser.add_argument("--val-json", default=str(DEFAULT_VAL_JSON))
    parser.add_argument("--test-json", default=str(DEFAULT_TEST_JSON))
    parser.add_argument("--inspection-summary", default=str(DEFAULT_INSPECTION_SUMMARY))
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    report_path = build_report(parse_args())
    print(report_path)


if __name__ == "__main__":
    main()
