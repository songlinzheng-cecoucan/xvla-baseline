#!/usr/bin/env python3
"""Evaluate SmolVLA flow-matching loss on held-out LeRobotDataset episodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

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
) -> dict[str, Any]:
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
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = preprocessor(batch)
            loss, output_dict = policy.forward(batch)
            batch_size_actual = int(next(iter(batch.values())).shape[0])
            loss_value = float(loss.item())
            losses.append(loss_value)
            total_loss += loss_value * batch_size_actual
            total_samples += batch_size_actual

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
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA validation flow-matching loss.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint pretrained_model directory.")
    parser.add_argument("--dataset-root", required=True, help="Local LeRobotDataset root.")
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo_id.")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode ids to evaluate.")
    parser.add_argument("--split-manifest", default=None, help="Episode split manifest JSON.")
    parser.add_argument("--split-name", default="val", choices=["train", "val", "test"], help="Split to use from manifest.")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap for smoke evaluation.")
    parser.add_argument("--device", default="cuda", help="Evaluation device.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--hf-datasets-cache", default="/tmp/hf_datasets_cache", help="Hugging Face datasets cache.")
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
    )

    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
