#!/usr/bin/env python3
"""Create episode-level train/val/test splits for a local LeRobotDataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_dataset_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"LeRobot meta/info.json not found: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def split_episode_ids(
    total_episodes: int,
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if total_episodes <= 0:
        raise ValueError("total_episodes must be positive")
    if train_ratio <= 0:
        raise ValueError("--train-ratio must be positive")
    if val_ratio < 0:
        raise ValueError("--val-ratio cannot be negative")
    if train_ratio + val_ratio > 1.0:
        raise ValueError("--train-ratio + --val-ratio must be <= 1.0")

    episodes = list(range(total_episodes))
    rng = random.Random(seed)
    rng.shuffle(episodes)

    if total_episodes == 1:
        return episodes, [], []

    train_count = int(round(total_episodes * train_ratio))
    val_count = int(round(total_episodes * val_ratio))

    train_count = max(1, min(train_count, total_episodes))
    remaining = total_episodes - train_count
    if val_ratio > 0 and remaining > 0:
        val_count = max(1, min(val_count, remaining))
    else:
        val_count = 0

    test_count = total_episodes - train_count - val_count
    if test_count < 0:
        val_count += test_count
        test_count = 0

    train_ids = sorted(episodes[:train_count])
    val_ids = sorted(episodes[train_count : train_count + val_count])
    test_ids = sorted(episodes[train_count + val_count :])
    return train_ids, val_ids, test_ids


def _split_explicit_episode_ids(
    episode_ids: list[int],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: str,
) -> tuple[list[int], list[int], list[int]]:
    if not episode_ids:
        raise ValueError("episode_ids must be non-empty")

    shuffled = list(episode_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        return sorted(shuffled), [], []

    train_count = int(round(len(shuffled) * train_ratio))
    val_count = int(round(len(shuffled) * val_ratio))

    train_count = max(1, min(train_count, len(shuffled)))
    remaining = len(shuffled) - train_count
    if val_ratio > 0 and remaining > 0:
        val_count = max(1, min(val_count, remaining))
    else:
        val_count = 0

    test_count = len(shuffled) - train_count - val_count
    if test_count < 0:
        val_count += test_count

    return (
        sorted(shuffled[:train_count]),
        sorted(shuffled[train_count : train_count + val_count]),
        sorted(shuffled[train_count + val_count :]),
    )


def load_task_distribution_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Task distribution manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError(f"Task distribution manifest has no tasks: {path}")
    return manifest


def split_episode_ids_by_task(
    task_distribution_manifest: dict[str, Any],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
    train_ids: list[int] = []
    val_ids: list[int] = []
    test_ids: list[int] = []
    task_splits: dict[str, Any] = {}

    for task_key, item in sorted(task_distribution_manifest["tasks"].items()):
        episode_ids = [int(value) for value in item.get("episode_indices", [])]
        if not episode_ids:
            raise ValueError(f"Task {task_key} has no episode_indices.")
        task_train, task_val, task_test = _split_explicit_episode_ids(
            episode_ids,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=f"{seed}:{task_key}",
        )
        train_ids.extend(task_train)
        val_ids.extend(task_val)
        test_ids.extend(task_test)
        task_splits[task_key] = {
            "train": task_train,
            "val": task_val,
            "test": task_test,
        }

    return sorted(train_ids), sorted(val_ids), sorted(test_ids), task_splits


def write_split_manifest(
    *,
    dataset_root: Path,
    dataset_repo_id: str,
    output_path: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    task_distribution_manifest_path: Path | None = None,
) -> dict[str, Any]:
    info = load_dataset_info(dataset_root)
    total_episodes = int(info["total_episodes"])
    task_splits = None
    if task_distribution_manifest_path is None:
        train_ids, val_ids, test_ids = split_episode_ids(
            total_episodes,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        split_type = "episode"
    else:
        task_distribution = load_task_distribution_manifest(task_distribution_manifest_path)
        train_ids, val_ids, test_ids, task_splits = split_episode_ids_by_task(
            task_distribution,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        split_type = "episode_stratified_by_task"

    all_ids = train_ids + val_ids + test_ids
    if sorted(all_ids) != list(range(total_episodes)):
        raise ValueError(
            "Split episode ids do not cover dataset episodes exactly once. "
            f"Expected 0..{total_episodes - 1}, got {len(set(all_ids))} unique ids."
        )

    manifest = {
        "dataset_root": str(dataset_root),
        "dataset_repo_id": dataset_repo_id,
        "seed": seed,
        "split_type": split_type,
        "total_episodes": total_episodes,
        "total_frames": int(info["total_frames"]),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "splits": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
    }
    if task_distribution_manifest_path is not None:
        manifest["task_distribution_manifest"] = str(task_distribution_manifest_path)
        manifest["task_splits"] = task_splits

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create episode-level split manifest for a LeRobotDataset.")
    parser.add_argument("--dataset-root", required=True, help="Local LeRobotDataset root.")
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo_id.")
    parser.add_argument("--output", required=True, help="Output split manifest JSON path.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Episode ratio for train split.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Episode ratio for validation split.")
    parser.add_argument("--seed", type=int, default=1000, help="Shuffle seed.")
    parser.add_argument(
        "--task-distribution-manifest",
        default=None,
        help="Optional task_distribution_manifest.json for stratified episode split.",
    )
    args = parser.parse_args()

    manifest = write_split_manifest(
        dataset_root=Path(args.dataset_root).expanduser(),
        dataset_repo_id=args.dataset_repo_id,
        output_path=Path(args.output).expanduser(),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        task_distribution_manifest_path=Path(args.task_distribution_manifest).expanduser()
        if args.task_distribution_manifest
        else None,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
