#!/usr/bin/env python3
"""Inspect SmolVLA single-step open-loop action predictions on LeRobotDataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_HF_DATASETS_CACHE = "/tmp/hf_datasets_cache"
PROXY_ENV_KEYS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
)

os.environ.setdefault("HF_DATASETS_CACHE", DEFAULT_HF_DATASETS_CACHE)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for _proxy_env_key in PROXY_ENV_KEYS:
    os.environ.pop(_proxy_env_key, None)

import numpy as np
import torch
from PIL import Image, ImageDraw

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors


ACTION_GROUPS = {
    "left_arm": slice(0, 7),
    "right_arm": slice(7, 14),
    "left_hand": slice(14, 20),
    "right_hand": slice(20, 26),
}
ACTION_DIM = 26


def prepare_offline_runtime() -> None:
    os.environ.setdefault("HF_DATASETS_CACHE", DEFAULT_HF_DATASETS_CACHE)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)


def parse_episode_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_split_episodes(path: str | None, split_name: str) -> list[int] | None:
    if path is None:
        return None
    manifest = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return [int(item) for item in manifest["splits"][split_name]]


def resolve_use_peft(value: str, checkpoint: Path) -> bool:
    if value == "auto":
        return (checkpoint / "adapter_config.json").exists()
    return value == "true"


def load_policy_processors(
    *,
    checkpoint: Path,
    dataset: LeRobotDataset,
    device: str,
    use_peft: bool,
):
    prepare_offline_runtime()
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint
    cfg.device = device
    cfg.use_peft = use_peft

    policy = make_policy(cfg=cfg, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": device},
        },
        postprocessor_overrides={
            "device_processor": {"device": "cpu"},
        },
    )
    return policy, preprocessor, postprocessor


def as_numpy_action(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float64)
    value = np.squeeze(value)
    if value.shape != (ACTION_DIM,):
        raise ValueError(f"Expected action shape ({ACTION_DIM},), got {value.shape}")
    return value


def sample_metadata(sample: dict[str, Any], dataset_index: int) -> dict[str, int]:
    metadata: dict[str, int] = {"dataset_index": int(dataset_index)}
    for key in ("episode_index", "frame_index", "timestamp"):
        if key not in sample:
            continue
        value = sample[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        if isinstance(value, np.generic):
            value = value.item()
        if key == "timestamp":
            metadata[key] = float(value)
        else:
            metadata[key] = int(value)
    return metadata


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    error = pred - gt
    abs_error = np.abs(error)
    squared_error = error * error
    metrics: dict[str, Any] = {
        "full_mae": float(abs_error.mean()),
        "full_mse": float(squared_error.mean()),
    }
    for group_name, group_slice in ACTION_GROUPS.items():
        metrics[f"{group_name}_mae"] = float(abs_error[group_slice].mean())
        metrics[f"{group_name}_mse"] = float(squared_error[group_slice].mean())
    return metrics


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def summarize(
    *,
    rows: list[dict[str, Any]],
    checkpoint: Path,
    dataset_root: Path,
    dataset_repo_id: str,
    split_name: str,
    episodes: list[int] | None,
    mode: str,
    use_peft: bool,
) -> dict[str, Any]:
    groups = {
        name: {
            "mae": mean_metric(rows, f"{name}_mae"),
            "mse": mean_metric(rows, f"{name}_mse"),
        }
        for name in ACTION_GROUPS
    }
    return {
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "dataset_repo_id": dataset_repo_id,
        "split_name": split_name,
        "episodes": episodes,
        "num_samples": len(rows),
        "mode": mode,
        "use_peft": use_peft,
        "metric_scope": "open-loop imitation sanity check; not XVLA benchmark success rate",
        "full_mae": mean_metric(rows, "full_mae"),
        "full_mse": mean_metric(rows, "full_mse"),
        "groups": groups,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_group_mae(path: Path, summary: dict[str, Any]) -> None:
    names = list(ACTION_GROUPS)
    values = [summary["groups"][name]["mae"] for name in names]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 90, 35, 60, 95
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    max_value = max(max(values), 1e-9)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin_left, 24), "Open-loop action MAE by group", fill="black")
    draw.line(
        [(margin_left, margin_top), (margin_left, margin_top + chart_height), (width - margin_right, margin_top + chart_height)],
        fill="#333333",
        width=2,
    )
    for tick in range(6):
        y = margin_top + chart_height - int(chart_height * tick / 5)
        value = max_value * tick / 5
        draw.line([(margin_left - 5, y), (width - margin_right, y)], fill="#e6e6e6")
        draw.text((8, y - 7), f"{value:.3f}", fill="#444444")

    bar_gap = 35
    bar_width = int((chart_width - bar_gap * (len(names) + 1)) / len(names))
    for index, (name, value) in enumerate(zip(names, values, strict=True)):
        x0 = margin_left + bar_gap + index * (bar_width + bar_gap)
        x1 = x0 + bar_width
        bar_height = int(chart_height * value / max_value)
        y0 = margin_top + chart_height - bar_height
        y1 = margin_top + chart_height
        draw.rectangle([(x0, y0), (x1, y1)], fill=colors[index])
        draw.text((x0, y1 + 12), name, fill="black")
        draw.text((x0, max(margin_top, y0 - 18)), f"{value:.4f}", fill="black")
    draw.text((18, margin_top + chart_height // 2), "MAE", fill="black")
    image.save(path)


def plot_error_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    series = {"full": [row["full_mae"] for row in rows]}
    series.update({name: [row[f"{name}_mae"] for row in rows] for name in ACTION_GROUPS})
    draw_line_chart(
        path=path,
        title="Open-loop action error over inspected samples",
        series=series,
        y_label="MAE",
        x_label="sample",
        width=1100,
        height=560,
    )


def plot_episode_action_compare(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    max_episodes: int,
    max_dims: int,
) -> None:
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        episode_index = record["metadata"].get("episode_index")
        if episode_index is not None:
            by_episode[int(episode_index)].append(record)

    for episode_index in sorted(by_episode)[:max_episodes]:
        episode_records = sorted(
            by_episode[episode_index],
            key=lambda item: item["metadata"].get("frame_index", item["metadata"]["dataset_index"]),
        )
        pred = np.stack([item["pred_action"] for item in episode_records])
        gt = np.stack([item["gt_action"] for item in episode_records])
        dims = min(max_dims, pred.shape[1])
        series = {}
        for dim in range(dims):
            series[f"gt_a{dim}"] = gt[:, dim].tolist()
            series[f"pred_a{dim}"] = pred[:, dim].tolist()
        draw_line_chart(
            path=output_dir / f"episode_{episode_index:06d}_action_compare.png",
            title=f"Episode {episode_index}: predicted vs GT action dims 0-{dims - 1}",
            series=series,
            y_label="action",
            x_label="inspected frame order",
            width=1200,
            height=max(520, 92 * dims),
        )


def draw_line_chart(
    *,
    path: Path,
    title: str,
    series: dict[str, list[float]],
    y_label: str,
    x_label: str,
    width: int,
    height: int,
) -> None:
    colors = [
        "#4c78a8",
        "#f58518",
        "#54a24b",
        "#e45756",
        "#72b7b2",
        "#b279a2",
        "#ff9da6",
        "#9d755d",
        "#bab0ac",
        "#000000",
    ]
    margin_left, margin_right, margin_top, margin_bottom = 90, 210, 60, 75
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    all_values = [value for values in series.values() for value in values]
    y_min = min(all_values) if all_values else 0.0
    y_max = max(all_values) if all_values else 1.0
    if abs(y_max - y_min) < 1e-9:
        y_min -= 1.0
        y_max += 1.0
    y_pad = 0.08 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad
    max_len = max((len(values) for values in series.values()), default=1)

    def x_pos(index: int) -> int:
        if max_len <= 1:
            return margin_left
        return margin_left + int(chart_width * index / (max_len - 1))

    def y_pos(value: float) -> int:
        return margin_top + chart_height - int(chart_height * (value - y_min) / (y_max - y_min))

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin_left, 24), title, fill="black")
    draw.line(
        [(margin_left, margin_top), (margin_left, margin_top + chart_height), (margin_left + chart_width, margin_top + chart_height)],
        fill="#333333",
        width=2,
    )
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_pos(value)
        draw.line([(margin_left - 5, y), (margin_left + chart_width, y)], fill="#e6e6e6")
        draw.text((8, y - 7), f"{value:.3f}", fill="#444444")

    for idx, (name, values) in enumerate(series.items()):
        if not values:
            continue
        points = [(x_pos(i), y_pos(float(value))) for i, value in enumerate(values)]
        color = colors[idx % len(colors)]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], fill=color)
        else:
            draw.line(points, fill=color, width=2)
        legend_y = margin_top + 18 * idx
        legend_x = margin_left + chart_width + 20
        draw.line([(legend_x, legend_y + 7), (legend_x + 24, legend_y + 7)], fill=color, width=3)
        draw.text((legend_x + 32, legend_y), name, fill="black")

    draw.text((18, margin_top + chart_height // 2), y_label, fill="black")
    draw.text((margin_left + chart_width // 2 - 25, height - 35), x_label, fill="black")
    image.save(path)


def inspect_open_loop(
    *,
    checkpoint: Path,
    dataset_root: Path,
    dataset_repo_id: str,
    episodes: list[int] | None,
    split_name: str,
    output_dir: Path,
    max_samples: int | None,
    device: str,
    use_peft: bool,
    plot_episodes: int,
    plot_dims: int,
) -> dict[str, Any]:
    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root, episodes=episodes)
    policy, preprocessor, postprocessor = load_policy_processors(
        checkpoint=checkpoint,
        dataset=dataset,
        device=device,
        use_peft=use_peft,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    with torch.inference_mode():
        for dataset_index in range(num_samples):
            sample = dataset[dataset_index]
            gt_action = as_numpy_action(sample["action"])

            policy_input = dict(sample)
            policy_input.pop("action", None)
            processed = preprocessor(policy_input)
            action_chunk = policy.predict_action_chunk(processed)
            raw_action = action_chunk[:, 0, :]
            pred_action = as_numpy_action(postprocessor(raw_action))

            metadata = sample_metadata(sample, dataset_index)
            metrics = compute_metrics(pred_action, gt_action)
            row = {**metadata, **metrics}
            rows.append(row)
            records.append(
                {
                    "metadata": metadata,
                    "pred_action": pred_action,
                    "gt_action": gt_action,
                    "metrics": metrics,
                }
            )

    if not rows:
        raise ValueError("No samples were inspected.")

    summary = summarize(
        rows=rows,
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        dataset_repo_id=dataset_repo_id,
        split_name=split_name,
        episodes=episodes,
        mode="single",
        use_peft=use_peft,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "per_group_metrics.json").write_text(
        json.dumps(summary["groups"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "per_sample_metrics.csv", rows)
    plot_group_mae(output_dir / "group_mae_bar.png", summary)
    plot_error_curve(output_dir / "action_error_curve.png", rows)
    plot_episode_action_compare(
        output_dir=output_dir,
        records=records,
        max_episodes=plot_episodes,
        max_dims=plot_dims,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SmolVLA open-loop single-step action predictions.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint pretrained_model directory.")
    parser.add_argument("--dataset-root", required=True, help="Local LeRobotDataset root.")
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo_id.")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode ids to inspect.")
    parser.add_argument("--split-manifest", default=None, help="Episode split manifest JSON.")
    parser.add_argument("--split-name", default="val", choices=["train", "val", "test"], help="Split from manifest.")
    parser.add_argument("--mode", default="single", choices=["single"], help="Only single-step mode is implemented in v1.")
    parser.add_argument("--max-samples", type=int, default=200, help="Maximum samples to inspect.")
    parser.add_argument("--device", default="cuda", help="Inference device.")
    parser.add_argument(
        "--use-peft",
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether to load checkpoint as PEFT/LoRA. auto detects adapter_config.json.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/CSV/PNG outputs.")
    parser.add_argument("--plot-episodes", type=int, default=2, help="Number of episodes to plot.")
    parser.add_argument("--plot-dims", type=int, default=8, help="Number of action dimensions to plot per episode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explicit_episodes = parse_episode_list(args.episodes)
    split_episodes = load_split_episodes(args.split_manifest, args.split_name)
    if explicit_episodes is not None and split_episodes is not None:
        raise ValueError("Use either --episodes or --split-manifest, not both.")
    episodes = explicit_episodes if explicit_episodes is not None else split_episodes

    checkpoint = Path(args.checkpoint).expanduser()
    use_peft = resolve_use_peft(args.use_peft, checkpoint)
    summary = inspect_open_loop(
        checkpoint=checkpoint,
        dataset_root=Path(args.dataset_root).expanduser(),
        dataset_repo_id=args.dataset_repo_id,
        episodes=episodes,
        split_name=args.split_name,
        output_dir=Path(args.output_dir).expanduser(),
        max_samples=args.max_samples,
        device=args.device,
        use_peft=use_peft,
        plot_episodes=args.plot_episodes,
        plot_dims=args.plot_dims,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
