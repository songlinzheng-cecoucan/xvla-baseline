#!/usr/bin/env python3
"""Build task distribution manifests from LeRobot conversion metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_task_categories(items: list[str]) -> dict[str, str]:
    categories: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Task category must be KEY=CATEGORY, got: {item}")
        key, category = item.split("=", 1)
        key = key.strip()
        category = category.strip()
        if not key or not category:
            raise ValueError(f"Task category must be KEY=CATEGORY, got: {item}")
        categories[key] = category
    return categories


def _task_key_for_source(source_path: Path, source_roots: list[Path], fallback_task: str) -> str:
    for root in source_roots:
        try:
            rel = source_path.relative_to(root)
        except ValueError:
            continue
        if rel.parts:
            return rel.parts[0]

    normalized = fallback_task.strip().lower().replace(",", " ").replace("/", " ")
    return "_".join(part for part in normalized.split() if part)


def build_task_distribution_manifest(
    *,
    dataset_root: Path,
    dataset_repo_id: str,
    conversion_manifest_path: Path,
    source_roots: list[Path],
    task_categories: dict[str, str],
) -> dict[str, Any]:
    info = _load_json(dataset_root / "meta" / "info.json")
    conversion = _load_json(conversion_manifest_path)
    episodes = conversion.get("episodes", [])
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"No converted episodes found in {conversion_manifest_path}")

    task_items: dict[str, dict[str, Any]] = {}
    task_text_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_dirs: dict[str, set[str]] = defaultdict(set)

    for episode in episodes:
        source_path = Path(episode["source_path"]).resolve()
        task_text = str(episode["task"])
        task_key = _task_key_for_source(source_path, source_roots, task_text)
        num_frames = int(episode["num_frames"])

        item = task_items.setdefault(
            task_key,
            {
                "num_episodes": 0,
                "num_frames": 0,
                "episode_indices": [],
            },
        )
        item["num_episodes"] += 1
        item["num_frames"] += num_frames
        item["episode_indices"].append(int(episode["episode_index"]))

        task_text_counts[task_key][task_text] += 1
        for root in source_roots:
            try:
                rel = source_path.relative_to(root)
            except ValueError:
                continue
            if rel.parts:
                source_dirs[task_key].add(str(root / rel.parts[0]))
                break

    for task_key, item in sorted(task_items.items()):
        item["task_texts"] = dict(sorted(task_text_counts[task_key].items()))
        item["source_dirs"] = sorted(source_dirs[task_key])
        if task_key in task_categories:
            item["xvla_relevance_category"] = task_categories[task_key]

    total_frames = sum(int(item["num_frames"]) for item in task_items.values())
    total_episodes = sum(int(item["num_episodes"]) for item in task_items.values())
    if total_episodes != int(info["total_episodes"]):
        raise ValueError(f"Episode count mismatch: manifest={total_episodes}, dataset={info['total_episodes']}")
    if total_frames != int(info["total_frames"]):
        raise ValueError(f"Frame count mismatch: manifest={total_frames}, dataset={info['total_frames']}")

    return {
        "dataset_root": str(dataset_root),
        "dataset_repo_id": dataset_repo_id,
        "source_roots": [str(root) for root in source_roots],
        "tasks": dict(sorted(task_items.items())),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "canonical_action_dim": 26,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write task distribution manifest for a converted Xsens dataset.")
    parser.add_argument("--dataset-root", required=True, help="Local LeRobotDataset root.")
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo_id.")
    parser.add_argument(
        "--conversion-manifest",
        default=None,
        help="Path to conversion_manifest.json. Defaults to DATASET_ROOT/conversion_manifest.json.",
    )
    parser.add_argument("--source-root", action="append", required=True, help="Source HDF5 root. May be repeated.")
    parser.add_argument("--task-category", action="append", default=[], help="Task relevance category as KEY=CATEGORY.")
    parser.add_argument("--output", required=True, help="Output manifest JSON path.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    conversion_manifest = (
        Path(args.conversion_manifest).expanduser().resolve()
        if args.conversion_manifest
        else dataset_root / "conversion_manifest.json"
    )
    manifest = build_task_distribution_manifest(
        dataset_root=dataset_root,
        dataset_repo_id=args.dataset_repo_id,
        conversion_manifest_path=conversion_manifest,
        source_roots=[Path(item).expanduser().resolve() for item in args.source_root],
        task_categories=_parse_task_categories(args.task_category),
    )
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
