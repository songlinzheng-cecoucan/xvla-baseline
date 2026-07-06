#!/usr/bin/env python3
"""Build a balanced multi-task LeRobotDataset from Xsens HDF5 roots."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from xvla_baseline.data.convert_xsens_to_lerobot import convert_xsens_to_lerobot, discover_hdf5_paths


def parse_task_roots(items: list[str]) -> dict[str, Path]:
    task_roots: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--task-root must be TASK_KEY=PATH, got: {item}")
        key, path = item.split("=", 1)
        key = key.strip()
        path = path.strip()
        if not key or not path:
            raise ValueError(f"--task-root must be TASK_KEY=PATH, got: {item}")
        if key in task_roots:
            raise ValueError(f"Duplicate task key: {key}")
        task_roots[key] = Path(path).expanduser().resolve()
    if not task_roots:
        raise ValueError("At least one --task-root is required.")
    return task_roots


def select_balanced_hdf5_paths(
    task_roots: dict[str, Path],
    *,
    episodes_per_task: int,
    seed: int,
) -> tuple[list[Path], dict[str, Any]]:
    if episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive.")

    selected_paths: list[Path] = []
    tasks: dict[str, Any] = {}
    for task_key, root in sorted(task_roots.items()):
        paths = discover_hdf5_paths([root])
        if len(paths) < episodes_per_task:
            raise ValueError(
                f"Task {task_key} has only {len(paths)} HDF5 trajectories, "
                f"but {episodes_per_task} were requested."
            )
        rng = random.Random(f"{seed}:{task_key}")
        selected = sorted(rng.sample(paths, episodes_per_task))
        start_episode_index = len(selected_paths)
        selected_paths.extend(selected)
        tasks[task_key] = {
            "source_root": str(root),
            "available_episodes": len(paths),
            "selected_episodes": len(selected),
            "episode_indices": list(range(start_episode_index, start_episode_index + len(selected))),
            "selected_paths": [str(path) for path in selected],
        }

    manifest = {
        "seed": seed,
        "episodes_per_task": episodes_per_task,
        "total_selected_episodes": len(selected_paths),
        "tasks": tasks,
    }
    return selected_paths, manifest


def write_selected_manifest(output_root: Path, manifest: dict[str, Any]) -> None:
    path = output_root / "selected_hdf5_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced Xsens multi-task LeRobotDataset.")
    parser.add_argument(
        "--task-root",
        action="append",
        required=True,
        help="Task root as TASK_KEY=PATH. May be repeated.",
    )
    parser.add_argument("--episodes-per-task", type=int, required=True, help="Number of episodes sampled per task.")
    parser.add_argument("--seed", type=int, default=1000, help="Deterministic sampling seed.")
    parser.add_argument("--output-root", required=True, help="Local LeRobotDataset output root.")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo_id metadata.")
    parser.add_argument("--fps", type=int, default=30, help="Dataset FPS.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output root if it already exists.")
    parser.add_argument("--max-frames-per-episode", type=int, default=None, help="Optional frame cap for smoke tests.")
    parser.add_argument("--use-videos", action="store_true", help="Encode image observations as videos.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print the deterministic selection summary; do not convert.",
    )
    args = parser.parse_args()

    task_roots = parse_task_roots(args.task_root)
    selected_paths, selected_manifest = select_balanced_hdf5_paths(
        task_roots,
        episodes_per_task=args.episodes_per_task,
        seed=args.seed,
    )

    if args.summary_only:
        print(json.dumps(selected_manifest, ensure_ascii=False, indent=2))
        return

    converted = convert_xsens_to_lerobot(
        selected_paths,
        output_root=args.output_root,
        repo_id=args.repo_id,
        fps=args.fps,
        overwrite=args.overwrite,
        max_frames_per_episode=args.max_frames_per_episode,
        use_videos=args.use_videos,
    )
    selected_manifest["converted_episodes"] = len(converted)
    selected_manifest["converted_frames"] = sum(item.num_frames for item in converted)
    write_selected_manifest(Path(args.output_root).expanduser(), selected_manifest)

    print(
        json.dumps(
            {
                "output_root": args.output_root,
                "repo_id": args.repo_id,
                "selected_episodes": len(selected_paths),
                "converted_episodes": len(converted),
                "converted_frames": selected_manifest["converted_frames"],
                "selected_manifest": str(Path(args.output_root).expanduser() / "selected_hdf5_manifest.json"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
