#!/usr/bin/env python3
"""Evaluate SmolVLA flow-matching loss on held-out LeRobotDataset episodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_HF_DATASETS_CACHE = "/tmp/hf_datasets_cache"

os.environ.setdefault("HF_DATASETS_CACHE", DEFAULT_HF_DATASETS_CACHE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors

PROXY_ENV_KEYS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
)


def parse_episode_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_split_episodes(path: str | None, split_name: str) -> list[int] | None:
    if path is None:
        return None
    manifest = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return [int(item) for item in manifest["splits"][split_name]]


def load_task_episode_lookup(path: Path) -> tuple[dict[int, str], dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(path.expanduser().read_text(encoding="utf-8"))
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError(f"Task distribution manifest has no tasks: {path}")

    episode_to_task: dict[int, str] = {}
    task_metadata: dict[str, dict[str, Any]] = {}
    for task_key, item in sorted(tasks.items()):
        episode_indices = [int(value) for value in item.get("episode_indices", [])]
        if not episode_indices:
            raise ValueError(f"Task {task_key} has no episode_indices in {path}")
        for episode_index in episode_indices:
            if episode_index in episode_to_task:
                previous = episode_to_task[episode_index]
                raise ValueError(
                    f"Episode {episode_index} appears in both task {previous} and task {task_key}."
                )
            episode_to_task[episode_index] = task_key
        task_metadata[task_key] = {
            "episode_indices": episode_indices,
            "num_episodes": int(item.get("num_episodes", len(episode_indices))),
            "num_frames": int(item.get("num_frames", 0)),
            "task_texts": item.get("task_texts", {}),
            "source_dirs": item.get("source_dirs", []),
            "xvla_relevance_category": item.get("xvla_relevance_category"),
        }
    return episode_to_task, task_metadata, manifest


def extract_episode_indices(batch: dict[str, Any]) -> list[int]:
    if "episode_index" not in batch:
        raise KeyError("Batch does not contain episode_index; cannot compute per-task metrics.")
    episode_index = batch["episode_index"]
    if hasattr(episode_index, "detach"):
        return [int(value) for value in episode_index.detach().cpu().view(-1).tolist()]
    if isinstance(episode_index, (list, tuple)):
        return [int(value) for value in episode_index]
    return [int(episode_index)]


def extract_batch_size(batch: dict[str, Any]) -> int:
    for value in batch.values():
        if hasattr(value, "shape") and len(value.shape) > 0:
            return int(value.shape[0])
    raise ValueError("Could not infer batch size from preprocessed batch.")


def load_policy_and_preprocessor(
    *,
    checkpoint: Path,
    dataset: LeRobotDataset,
    device: str,
    use_peft: bool,
):
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint
    cfg.device = device
    cfg.use_peft = use_peft

    policy = make_policy(cfg=cfg, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": device},
        },
    )
    return policy, preprocessor


def evaluate_dataset_batches(
    *,
    dataset: LeRobotDataset,
    policy: torch.nn.Module,
    preprocessor: Any,
    batch_size: int,
    max_batches: int | None,
    num_workers: int,
    track_episodes: bool,
    progress_every: int | None,
    progress_label: str,
) -> dict[str, Any]:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    total_loss = 0.0
    total_samples = 0
    losses: list[float] = []
    episodes: set[int] = set()
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            if track_episodes:
                episodes.update(extract_episode_indices(batch))
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            batch_size_actual = extract_batch_size(batch)
            loss_value = float(loss.item())
            losses.append(loss_value)
            total_loss += loss_value * batch_size_actual
            total_samples += batch_size_actual
            if progress_every and progress_every > 0 and (batch_index + 1) % progress_every == 0:
                print(
                    f"[eval:{progress_label}] batches={batch_index + 1} "
                    f"samples={total_samples} latest_loss={loss_value:.6f}",
                    flush=True,
                )

    return {
        "total_loss": total_loss,
        "total_samples": total_samples,
        "num_batches": len(losses),
        "batch_losses": losses,
        "episodes": episodes,
    }


def evaluate_loss(
    *,
    checkpoint: Path,
    dataset_root: Path,
    dataset_repo_id: str,
    episodes: list[int] | None,
    batch_size: int,
    max_batches: int | None,
    device: str,
    num_workers: int,
    use_peft: bool | None,
    task_distribution_manifest: Path | None,
    per_task: bool,
    progress_every: int | None,
) -> dict[str, Any]:
    if per_task and task_distribution_manifest is None:
        raise ValueError("--per-task requires --task-distribution-manifest.")

    episode_to_task: dict[int, str] = {}
    task_metadata: dict[str, dict[str, Any]] = {}
    raw_task_manifest: dict[str, Any] | None = None
    if task_distribution_manifest is not None:
        episode_to_task, task_metadata, raw_task_manifest = load_task_episode_lookup(task_distribution_manifest)

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint
    cfg.device = device
    if use_peft is None:
        use_peft = (checkpoint / "adapter_config.json").exists()
    cfg.use_peft = use_peft

    meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(cfg, meta)
    dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
    )
    policy, preprocessor = load_policy_and_preprocessor(
        checkpoint=checkpoint,
        dataset=dataset,
        device=device,
        use_peft=use_peft,
    )

    total_loss = 0.0
    total_samples = 0
    losses: list[float] = []
    task_accumulators: dict[str, dict[str, Any]] = {}
    if per_task:
        selected_episodes = set(episodes) if episodes is not None else set(episode_to_task)
        for task_key, metadata in task_metadata.items():
            task_episodes = sorted(
                episode_index
                for episode_index in metadata["episode_indices"]
                if episode_index in selected_episodes
            )
            if task_episodes:
                task_accumulators[task_key] = {
                    "total_loss": 0.0,
                    "num_samples": 0,
                    "num_batches": 0,
                    "episodes": set(),
                    "split_episodes": task_episodes,
                }

    if per_task:
        remaining_batches = max_batches
        for task_key, accumulator in sorted(task_accumulators.items()):
            if remaining_batches is not None and remaining_batches <= 0:
                break
            task_dataset = LeRobotDataset(
                dataset_repo_id,
                root=dataset_root,
                episodes=accumulator["split_episodes"],
                delta_timestamps=delta_timestamps,
            )
            task_result = evaluate_dataset_batches(
                dataset=task_dataset,
                policy=policy,
                preprocessor=preprocessor,
                batch_size=batch_size,
                max_batches=remaining_batches,
                num_workers=num_workers,
                track_episodes=True,
                progress_every=progress_every,
                progress_label=task_key,
            )
            accumulator["total_loss"] += task_result["total_loss"]
            accumulator["num_samples"] += task_result["total_samples"]
            accumulator["num_batches"] += task_result["num_batches"]
            accumulator["episodes"].update(task_result["episodes"])
            total_loss += task_result["total_loss"]
            total_samples += task_result["total_samples"]
            losses.extend(task_result["batch_losses"])
            if remaining_batches is not None:
                remaining_batches -= int(task_result["num_batches"])
    else:
        eval_result = evaluate_dataset_batches(
            dataset=dataset,
            policy=policy,
            preprocessor=preprocessor,
            batch_size=batch_size,
            max_batches=max_batches,
            num_workers=num_workers,
            track_episodes=False,
            progress_every=progress_every,
            progress_label="global",
        )
        total_loss = eval_result["total_loss"]
        total_samples = eval_result["total_samples"]
        losses = eval_result["batch_losses"]

    if total_samples <= 0:
        raise ValueError("No validation samples were evaluated.")

    result = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "dataset_repo_id": dataset_repo_id,
        "episodes": episodes,
        "num_samples": total_samples,
        "num_batches": len(losses),
        "batch_size": batch_size,
        "mean_loss": total_loss / total_samples,
        "batch_losses": losses,
        "device": device,
        "use_peft": use_peft,
    }
    if task_distribution_manifest is not None:
        result["task_distribution_manifest"] = str(task_distribution_manifest)
    if raw_task_manifest is not None:
        result["task_manifest_total_episodes"] = raw_task_manifest.get("total_episodes")
        result["task_manifest_total_frames"] = raw_task_manifest.get("total_frames")
    if per_task:
        per_task_result: dict[str, Any] = {}
        for task_key, accumulator in sorted(task_accumulators.items()):
            num_samples = int(accumulator["num_samples"])
            metadata = task_metadata.get(task_key, {})
            per_task_result[task_key] = {
                "mean_loss": accumulator["total_loss"] / num_samples if num_samples else None,
                "num_samples": num_samples,
                "num_batches": int(accumulator["num_batches"]),
                "episodes": sorted(int(value) for value in accumulator["episodes"]),
                "split_episodes": [int(value) for value in accumulator["split_episodes"]],
                "num_split_episodes": len(accumulator["split_episodes"]),
                "task_texts": metadata.get("task_texts", {}),
                "xvla_relevance_category": metadata.get("xvla_relevance_category"),
            }
        result["per_task"] = per_task_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA validation flow-matching loss.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint pretrained_model directory.")
    parser.add_argument("--dataset-root", required=True, help="Local LeRobotDataset root.")
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo_id.")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode ids to evaluate.")
    parser.add_argument("--split-manifest", default=None, help="Episode split manifest JSON.")
    parser.add_argument("--split-name", default="val", choices=["train", "val", "test"], help="Split to use from manifest.")
    parser.add_argument("--task-distribution-manifest", default=None, help="Task distribution manifest JSON.")
    parser.add_argument("--per-task", action="store_true", help="Aggregate loss by task using task distribution manifest.")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap for smoke evaluation.")
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N batches. 0 disables it.")
    parser.add_argument("--device", default="cuda", help="Evaluation device.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--hf-datasets-cache", default=DEFAULT_HF_DATASETS_CACHE, help="Hugging Face datasets cache.")
    parser.add_argument("--online", dest="offline", action="store_false", help="Allow Hugging Face Hub network access.")
    parser.set_defaults(offline=True)
    parser.add_argument("--keep-proxy", dest="clear_proxy", action="store_false", help="Keep proxy env vars.")
    parser.set_defaults(clear_proxy=True)
    parser.add_argument(
        "--use-peft",
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether to load checkpoint as a PEFT/LoRA adapter. auto detects adapter_config.json.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    explicit_episodes = parse_episode_list(args.episodes)
    split_episodes = load_split_episodes(args.split_manifest, args.split_name)
    if explicit_episodes is not None and split_episodes is not None:
        raise ValueError("Use either --episodes or --split-manifest, not both.")
    episodes = explicit_episodes if explicit_episodes is not None else split_episodes
    use_peft = None if args.use_peft == "auto" else args.use_peft == "true"
    if args.clear_proxy:
        for key in PROXY_ENV_KEYS:
            os.environ.pop(key, None)
    os.environ.setdefault("HF_DATASETS_CACHE", args.hf_datasets_cache)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    result = evaluate_loss(
        checkpoint=Path(args.checkpoint).expanduser(),
        dataset_root=Path(args.dataset_root).expanduser(),
        dataset_repo_id=args.dataset_repo_id,
        episodes=episodes,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device=args.device,
        num_workers=args.num_workers,
        use_peft=use_peft,
        task_distribution_manifest=Path(args.task_distribution_manifest).expanduser()
        if args.task_distribution_manifest
        else None,
        per_task=args.per_task,
        progress_every=args.progress_every or None,
    )
    if args.split_manifest:
        result["split_manifest"] = str(Path(args.split_manifest).expanduser())
        result["split_name"] = args.split_name

    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
