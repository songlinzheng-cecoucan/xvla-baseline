#!/usr/bin/env python3
"""Convert RoboMIND TienKung Xsens HDF5 trajectories to a LeRobotDataset.

The converter is intentionally narrow for the XVLA baseline smoke path:

    observation.images.camera_top: RGB image, HWC uint8
    observation.state:            canonical 26D float32
    action:                       canonical 26D float32
    task:                         HDF5 language_raw[0]
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from xvla_baseline.data.robomind_xsens_reader import (
    XsensTrajectory,
    decode_rgb_frame,
    read_xsens_trajectory,
)


DEFAULT_FPS = 30
DEFAULT_REPO_ID = "local/robomind_tienkung_xsens_smoke"
ROBOT_TYPE = "tienkung_xsens"
IMAGE_FEATURE_KEY = "observation.images.camera_top"
STATE_FEATURE_KEY = "observation.state"
ACTION_FEATURE_KEY = "action"

CANONICAL_26_NAMES = [
    "left_arm_0",
    "left_arm_1",
    "left_arm_2",
    "left_arm_3",
    "left_arm_4",
    "left_arm_5",
    "left_arm_6",
    "right_arm_0",
    "right_arm_1",
    "right_arm_2",
    "right_arm_3",
    "right_arm_4",
    "right_arm_5",
    "right_arm_6",
    "left_hand_0",
    "left_hand_1",
    "left_hand_2",
    "left_hand_3",
    "left_hand_4",
    "left_hand_5",
    "right_hand_0",
    "right_hand_1",
    "right_hand_2",
    "right_hand_3",
    "right_hand_4",
    "right_hand_5",
]


@dataclass
class ConvertedEpisode:
    episode_index: int
    source_path: str
    task: str
    num_frames: int
    state_shape: tuple[int, int]
    action_shape: tuple[int, int]
    rgb_camera_top_size: tuple[int, int]


def discover_hdf5_paths(inputs: Iterable[str | Path]) -> list[Path]:
    """Resolve explicit HDF5 files and recursively discover trajectory.hdf5 files."""

    paths: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.rglob("trajectory.hdf5")))
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")

    unique_paths = sorted({path.resolve() for path in paths})
    if not unique_paths:
        raise ValueError("No HDF5 trajectory files were found.")
    return unique_paths


def make_features(image_size: tuple[int, int]) -> dict[str, dict[str, Any]]:
    width, height = image_size
    return {
        IMAGE_FEATURE_KEY: {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        STATE_FEATURE_KEY: {
            "dtype": "float32",
            "shape": (26,),
            "names": CANONICAL_26_NAMES,
        },
        ACTION_FEATURE_KEY: {
            "dtype": "float32",
            "shape": (26,),
            "names": CANONICAL_26_NAMES,
        },
    }


def _require_consistent_image_size(trajectory: XsensTrajectory, expected_size: tuple[int, int]) -> None:
    image_size = tuple(trajectory.metadata["rgb_camera_top_size"])
    if image_size != expected_size:
        raise ValueError(
            "All trajectories in one LeRobotDataset must have the same image size. "
            f"Expected {expected_size}, got {image_size} for {trajectory.path}"
        )


def convert_xsens_to_lerobot(
    hdf5_paths: list[Path],
    *,
    output_root: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    fps: int = DEFAULT_FPS,
    overwrite: bool = False,
    max_frames_per_episode: int | None = None,
    use_videos: bool = False,
) -> list[ConvertedEpisode]:
    """Convert HDF5 trajectories into a local LeRobotDataset."""

    output_path = Path(output_root).expanduser()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_path}. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    first = read_xsens_trajectory(hdf5_paths[0], decode_images=False)
    image_size = tuple(first.metadata["rgb_camera_top_size"])
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=make_features(image_size),
        root=output_path,
        robot_type=ROBOT_TYPE,
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=4,
    )

    converted: list[ConvertedEpisode] = []
    for episode_index, hdf5_path in enumerate(hdf5_paths):
        trajectory = first if episode_index == 0 else read_xsens_trajectory(hdf5_path, decode_images=False)
        _require_consistent_image_size(trajectory, image_size)

        frame_count = trajectory.num_frames
        if max_frames_per_episode is not None:
            frame_count = min(frame_count, max_frames_per_episode)

        for frame_index in range(frame_count):
            dataset.add_frame(
                {
                    IMAGE_FEATURE_KEY: decode_rgb_frame(trajectory.rgb_camera_top[frame_index]),
                    STATE_FEATURE_KEY: trajectory.state[frame_index].astype(np.float32, copy=False),
                    ACTION_FEATURE_KEY: trajectory.action[frame_index].astype(np.float32, copy=False),
                    "task": trajectory.task,
                }
            )
        dataset.save_episode()

        converted.append(
            ConvertedEpisode(
                episode_index=episode_index,
                source_path=str(trajectory.path),
                task=trajectory.task,
                num_frames=frame_count,
                state_shape=(frame_count, 26),
                action_shape=(frame_count, 26),
                rgb_camera_top_size=image_size,
            )
        )

    dataset.finalize()
    write_manifest(output_path, repo_id=repo_id, fps=fps, use_videos=use_videos, converted=converted)
    return converted


def write_manifest(
    output_root: Path,
    *,
    repo_id: str,
    fps: int,
    use_videos: bool,
    converted: list[ConvertedEpisode],
) -> None:
    manifest = {
        "repo_id": repo_id,
        "robot_type": ROBOT_TYPE,
        "fps": fps,
        "use_videos": use_videos,
        "features": {
            IMAGE_FEATURE_KEY: "RGB HWC uint8 image from observations/rgb_images/camera_top",
            STATE_FEATURE_KEY: "canonical 26D puppet absolute state",
            ACTION_FEATURE_KEY: "canonical 26D puppet absolute action",
        },
        "episodes": [asdict(item) for item in converted],
    }
    (output_root / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RoboMIND TienKung Xsens HDF5 files to LeRobotDataset.")
    parser.add_argument("inputs", nargs="+", help="HDF5 files or directories containing trajectory.hdf5 files.")
    parser.add_argument("--output-root", required=True, help="Local LeRobotDataset output root.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot dataset repo_id metadata.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Dataset FPS. Defaults to 30 for RoboMIND smoke data.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output root if it already exists.")
    parser.add_argument("--max-frames-per-episode", type=int, default=None, help="Optional frame cap for smoke tests.")
    parser.add_argument("--use-videos", action="store_true", help="Encode image observations as videos instead of PNG files.")
    args = parser.parse_args()

    hdf5_paths = discover_hdf5_paths(args.inputs)
    converted = convert_xsens_to_lerobot(
        hdf5_paths,
        output_root=args.output_root,
        repo_id=args.repo_id,
        fps=args.fps,
        overwrite=args.overwrite,
        max_frames_per_episode=args.max_frames_per_episode,
        use_videos=args.use_videos,
    )
    print(json.dumps({"output_root": args.output_root, "episodes": converted}, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
