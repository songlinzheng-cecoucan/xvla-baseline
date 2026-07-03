#!/usr/bin/env python3
"""Train SmolVLA with LoRA on a converted RoboMIND Xsens LeRobotDataset.

This wrapper keeps the XVLA baseline-specific LeRobot CLI overrides in one
place. It intentionally shells out to `conda run -n lerobot312 lerobot-train`
so it can be launched from the repo without manually activating the training
environment first.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


DEFAULT_DATASET_ROOT = "/tmp/robomind_xsens_lerobot_smoke"
DEFAULT_REPO_ID = "local/robomind_tienkung_xsens_smoke"
DEFAULT_OUTPUT_DIR = "/tmp/robomind_smolvla_lora_train"
DEFAULT_POLICY_PATH = "lerobot/smolvla_base"
DEFAULT_CONDA_ENV = "lerobot312"

PROXY_ENV_KEYS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
)


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "--no-capture-output",
        "lerobot-train",
        f"--policy.path={args.policy_path}",
        "--policy.push_to_hub=false",
        "--policy.input_features=null",
        "--policy.output_features=null",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={args.dataset_root}",
        f"--output_dir={args.output_dir}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--eval_freq={args.eval_freq}",
        f"--num_workers={args.num_workers}",
        "--peft.method_type=LORA",
        f"--peft.r={args.lora_rank}",
        f"--wandb.enable={str(args.wandb).lower()}",
    ]
    episodes = resolve_episodes(args)
    if episodes is not None:
        command.append(f"--dataset.episodes={json.dumps(episodes)}")
    return command


def resolve_episodes(args: argparse.Namespace) -> list[int] | None:
    if args.episodes and args.split_manifest:
        raise ValueError("Use either --episodes or --split-manifest, not both.")
    if args.episodes:
        return [int(item.strip()) for item in args.episodes.split(",") if item.strip()]
    if args.split_manifest:
        manifest = json.loads(Path(args.split_manifest).expanduser().read_text(encoding="utf-8"))
        episodes = manifest["splits"][args.split_name]
        return [int(item) for item in episodes]
    return None


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.clear_proxy:
        for key in PROXY_ENV_KEYS:
            env.pop(key, None)

    env.setdefault("HF_DATASETS_CACHE", args.hf_datasets_cache)
    env.setdefault("WANDB_MODE", "disabled")
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
    return env


def validate_args(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).expanduser()
    if not args.dry_run and not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.lora_rank <= 0:
        raise ValueError("--lora-rank must be positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SmolVLA LoRA training for the XVLA baseline.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Converted LeRobotDataset root.")
    parser.add_argument("--dataset-repo-id", default=DEFAULT_REPO_ID, help="LeRobot dataset repo_id.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Training output directory.")
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH, help="Base SmolVLA policy path.")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV, help="Conda env containing LeRobot.")
    parser.add_argument("--steps", type=int, default=1, help="Training update steps.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-process batch size.")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--save-freq", type=int, default=1, help="Checkpoint frequency in steps.")
    parser.add_argument("--log-freq", type=int, default=1, help="Logging frequency in steps.")
    parser.add_argument("--eval-freq", type=int, default=0, help="Evaluation frequency. 0 disables eval.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode ids to train on, e.g. 0,1,2.")
    parser.add_argument("--split-manifest", default=None, help="Episode split manifest JSON.")
    parser.add_argument("--split-name", default="train", choices=["train", "val", "test"], help="Split to use from manifest.")
    parser.add_argument("--hf-datasets-cache", default="/tmp/hf_datasets_cache", help="Hugging Face datasets cache.")
    parser.add_argument("--online", dest="offline", action="store_false", help="Allow Hugging Face Hub network access.")
    parser.set_defaults(offline=True)
    parser.add_argument("--keep-proxy", dest="clear_proxy", action="store_false", help="Keep proxy env vars.")
    parser.set_defaults(clear_proxy=True)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    command = build_command(args)

    print("Running command:")
    print(" ".join(command))

    if args.dry_run:
        return

    subprocess.run(command, env=build_env(args), check=True)


if __name__ == "__main__":
    main()
