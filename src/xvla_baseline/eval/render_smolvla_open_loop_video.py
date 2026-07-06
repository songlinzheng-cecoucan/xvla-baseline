#!/usr/bin/env python3
"""Render an open-loop SmolVLA action comparison video for one episode."""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
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

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from xvla_baseline.eval.inspect_smolvla_open_loop import (
    ACTION_DIM,
    ACTION_GROUPS,
    as_numpy_action,
    compute_metrics,
    load_policy_processors,
    resolve_use_peft,
)


def tensor_image_to_pil(value: torch.Tensor | np.ndarray) -> Image.Image:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"Expected image ndim 3, got {array.shape}")
    if array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array[..., :3], mode="RGB")


def sample_scalar(sample: dict[str, Any], key: str, default: int | float = 0) -> int | float:
    if key not in sample:
        return default
    value = sample[key]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    if isinstance(value, np.generic):
        value = value.item()
    return value


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str = "black") -> None:
    draw.text(xy, text, fill=fill)


def draw_metric_bars(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    metrics: dict[str, float],
    max_value: float,
) -> None:
    colors = {
        "full": "#333333",
        "left_arm": "#4c78a8",
        "right_arm": "#f58518",
        "left_hand": "#54a24b",
        "right_hand": "#e45756",
    }
    row_h = 30
    for idx, (name, value) in enumerate(metrics.items()):
        yy = y + idx * row_h
        draw_text(draw, (x, yy), f"{name:>10}", fill="black")
        bar_x = x + 92
        bar_w = int(width * min(value / max_value, 1.0)) if max_value > 0 else 0
        draw.rectangle([(bar_x, yy + 3), (bar_x + width, yy + 18)], outline="#cccccc")
        draw.rectangle([(bar_x, yy + 3), (bar_x + bar_w, yy + 18)], fill=colors[name])
        draw_text(draw, (bar_x + width + 10, yy), f"{value:.4f}", fill="black")


def draw_action_strip(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    pred: np.ndarray,
    gt: np.ndarray,
) -> None:
    values = np.concatenate([pred, gt])
    value_min = float(values.min())
    value_max = float(values.max())
    if abs(value_max - value_min) < 1e-9:
        value_min -= 1.0
        value_max += 1.0
    zero_y = y + height - int(height * (0.0 - value_min) / (value_max - value_min))
    zero_y = max(y, min(y + height, zero_y))
    draw.rectangle([(x, y), (x + width, y + height)], outline="#bbbbbb")
    draw.line([(x, zero_y), (x + width, zero_y)], fill="#dddddd")
    group_edges = [7, 14, 20]
    bar_gap = 2
    bar_w = max(3, int((width - (ACTION_DIM - 1) * bar_gap) / ACTION_DIM))
    for dim in range(ACTION_DIM):
        xx = x + dim * (bar_w + bar_gap)
        for value, color, offset in ((gt[dim], "#999999", 0), (pred[dim], "#e45756", int(bar_w * 0.45))):
            yy = y + height - int(height * (float(value) - value_min) / (value_max - value_min))
            draw.rectangle(
                [(xx + offset, min(zero_y, yy)), (xx + offset + max(1, int(bar_w * 0.4)), max(zero_y, yy))],
                fill=color,
            )
    for edge in group_edges:
        xx = x + edge * (bar_w + bar_gap) - bar_gap
        draw.line([(xx, y), (xx, y + height)], fill="#444444")
    draw_text(draw, (x, y - 20), "26D action: gray=GT, red=pred", fill="black")


def draw_error_curve(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    history: list[float],
) -> None:
    draw.rectangle([(x, y), (x + width, y + height)], outline="#bbbbbb")
    if len(history) < 2:
        return
    max_value = max(max(history), 1e-9)
    points = []
    for idx, value in enumerate(history):
        xx = x + int(width * idx / (len(history) - 1))
        yy = y + height - int(height * value / max_value)
        points.append((xx, yy))
    draw.line(points, fill="#4c78a8", width=2)
    draw_text(draw, (x, y - 20), f"full MAE history, max={max_value:.4f}", fill="black")


def render_frame(
    *,
    image: Image.Image,
    pred: np.ndarray,
    gt: np.ndarray,
    metrics: dict[str, Any],
    frame_index: int,
    episode_index: int,
    task: str,
    history: list[float],
    canvas_size: tuple[int, int],
) -> np.ndarray:
    canvas_w, canvas_h = canvas_size
    canvas = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(canvas)

    image_h = canvas_h
    image_w = int(image.width * image_h / image.height)
    image_w = min(image_w, canvas_w // 2)
    image_resized = image.resize((image_w, image_h))
    canvas.paste(image_resized, (0, 0))

    panel_x = image_w + 24
    draw_text(draw, (panel_x, 22), f"Episode {episode_index}  Frame {frame_index}", fill="black")
    draw_text(draw, (panel_x, 46), f"Task: {task}", fill="black")
    draw_text(draw, (panel_x, 78), "Open-loop single-step action comparison", fill="#444444")
    draw_text(draw, (panel_x, 102), "Not a closed-loop XVLA benchmark rollout", fill="#a00000")

    mae_metrics = {
        "full": float(metrics["full_mae"]),
        "left_arm": float(metrics["left_arm_mae"]),
        "right_arm": float(metrics["right_arm_mae"]),
        "left_hand": float(metrics["left_hand_mae"]),
        "right_hand": float(metrics["right_hand_mae"]),
    }
    max_mae = max(max(mae_metrics.values()) * 1.15, 1e-6)
    draw_metric_bars(draw, x=panel_x, y=145, width=260, metrics=mae_metrics, max_value=max_mae)

    strip_y = 330
    strip_w = canvas_w - panel_x - 38
    draw_action_strip(draw, x=panel_x, y=strip_y, width=strip_w, height=145, pred=pred, gt=gt)
    draw_error_curve(draw, x=panel_x, y=535, width=strip_w, height=120, history=history)

    array = np.asarray(canvas)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def render_video(
    *,
    checkpoint: Path,
    dataset_root: Path,
    dataset_repo_id: str,
    episode: int,
    output_path: Path,
    max_frames: int,
    stride: int,
    fps: int,
    device: str,
    use_peft: bool,
    canvas_size: tuple[int, int],
) -> dict[str, Any]:
    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root, episodes=[episode])
    policy, preprocessor, postprocessor = load_policy_processors(
        checkpoint=checkpoint,
        dataset=dataset,
        device=device,
        use_peft=use_peft,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), canvas_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    frame_count = min(len(dataset), max_frames * stride)
    history: deque[float] = deque(maxlen=120)
    rows = []

    with torch.inference_mode():
        for local_index in range(0, frame_count, stride):
            sample = dataset[local_index]
            gt_action = as_numpy_action(sample["action"])
            policy_input = dict(sample)
            policy_input.pop("action", None)
            processed = preprocessor(policy_input)
            action_chunk = policy.predict_action_chunk(processed)
            pred_action = as_numpy_action(postprocessor(action_chunk[:, 0, :]))

            metrics = compute_metrics(pred_action, gt_action)
            history.append(float(metrics["full_mae"]))
            frame_index = int(sample_scalar(sample, "frame_index", local_index))
            task = str(sample.get("task", ""))
            pil_image = tensor_image_to_pil(sample["observation.images.camera_top"])
            frame = render_frame(
                image=pil_image,
                pred=pred_action,
                gt=gt_action,
                metrics=metrics,
                frame_index=frame_index,
                episode_index=episode,
                task=task,
                history=list(history),
                canvas_size=canvas_size,
            )
            writer.write(frame)
            rows.append(metrics)

    writer.release()
    summary = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "dataset_repo_id": dataset_repo_id,
        "episode": episode,
        "output_path": str(output_path),
        "num_video_frames": len(rows),
        "stride": stride,
        "fps": fps,
        "mode": "single",
        "metric_scope": "open-loop imitation visualization; not XVLA benchmark success rate",
        "full_mae": float(np.mean([row["full_mae"] for row in rows])) if rows else None,
        "groups": {
            name: {
                "mae": float(np.mean([row[f"{name}_mae"] for row in rows])) if rows else None,
                "mse": float(np.mean([row[f"{name}_mse"] for row in rows])) if rows else None,
            }
            for name in ACTION_GROUPS
        },
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SmolVLA open-loop action comparison video.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint pretrained_model directory.")
    parser.add_argument("--dataset-root", required=True, help="Local LeRobotDataset root.")
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo_id.")
    parser.add_argument("--episode", type=int, required=True, help="Episode id to render.")
    parser.add_argument("--output", required=True, help="Output .mp4 path.")
    parser.add_argument("--max-frames", type=int, default=120, help="Maximum rendered frames.")
    parser.add_argument("--stride", type=int, default=2, help="Dataset frame stride.")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS.")
    parser.add_argument("--device", default="cuda", help="Inference device.")
    parser.add_argument(
        "--use-peft",
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether to load checkpoint as PEFT/LoRA. auto detects adapter_config.json.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Output video width.")
    parser.add_argument("--height", type=int, default=720, help="Output video height.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser()
    summary = render_video(
        checkpoint=checkpoint,
        dataset_root=Path(args.dataset_root).expanduser(),
        dataset_repo_id=args.dataset_repo_id,
        episode=args.episode,
        output_path=Path(args.output).expanduser(),
        max_frames=args.max_frames,
        stride=args.stride,
        fps=args.fps,
        device=args.device,
        use_peft=resolve_use_peft(args.use_peft, checkpoint),
        canvas_size=(args.width, args.height),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
