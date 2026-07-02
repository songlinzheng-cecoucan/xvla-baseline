#!/usr/bin/env python3
"""Export XVLA baseline JSON metrics to TensorBoard event files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


STEP_RE = re.compile(r"_(\d{6})_")


def infer_step(path: Path) -> int:
    match = STEP_RE.search(path.name)
    if match:
        return int(match.group(1))
    data = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = str(data.get("checkpoint", ""))
    match = STEP_RE.search(checkpoint)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not infer step from {path}")


def export_loss_json(writer: SummaryWriter, path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    step = infer_step(path)
    split = "test" if path.name.startswith("test_") else "val"
    max_batches = data.get("num_batches")
    suffix = f"max{max_batches}" if max_batches is not None else "all"
    writer.add_scalar(f"{split}/flow_matching_loss_{suffix}", float(data["mean_loss"]), step)
    writer.add_scalar(f"{split}/num_samples_{suffix}", int(data["num_samples"]), step)


def export_open_loop_summary(writer: SummaryWriter, path: Path, step: int | None) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if step is None:
        step = infer_step(path)
    split = data.get("split_name", "unknown")
    writer.add_scalar(f"{split}/open_loop/full_mae", float(data["full_mae"]), step)
    writer.add_scalar(f"{split}/open_loop/full_mse", float(data["full_mse"]), step)
    for group_name, group_metrics in data.get("groups", {}).items():
        writer.add_scalar(f"{split}/open_loop/{group_name}_mae", float(group_metrics["mae"]), step)
        writer.add_scalar(f"{split}/open_loop/{group_name}_mse", float(group_metrics["mse"]), step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export baseline JSON metrics to TensorBoard.")
    parser.add_argument("--run-dir", required=True, help="Directory containing val_loss/test_loss JSON files.")
    parser.add_argument("--log-dir", required=True, help="TensorBoard log directory to write.")
    parser.add_argument(
        "--open-loop-summary",
        default=None,
        help="Optional open-loop summary.json path to export.",
    )
    parser.add_argument(
        "--open-loop-step",
        type=int,
        default=None,
        help="Step for open-loop metrics when it cannot be inferred from the summary path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser()
    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    metric_paths = sorted(run_dir.glob("val_loss_*_max*.json")) + sorted(
        run_dir.glob("test_loss_*_max*.json")
    )
    if not metric_paths and args.open_loop_summary is None:
        raise ValueError(f"No metric JSON files found in {run_dir}")

    with SummaryWriter(log_dir=str(log_dir)) as writer:
        for path in metric_paths:
            export_loss_json(writer, path)
        if args.open_loop_summary:
            export_open_loop_summary(writer, Path(args.open_loop_summary).expanduser(), args.open_loop_step)
        writer.flush()

    print(f"Wrote TensorBoard events to {log_dir}")


if __name__ == "__main__":
    main()
