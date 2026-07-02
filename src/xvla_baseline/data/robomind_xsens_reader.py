#!/usr/bin/env python3
"""Reader for RoboMIND TienKung Xsens 1RGB trajectories.

The v1 reader intentionally supports only the documented Xsens schema:

    puppet/joint_position: (T, 14)
    puppet/end_effector:   (T, 12)
    observations/rgb_images/camera_top: (T,)

It maps those fields into the canonical XVLA 26D order:

    left_arm[7], right_arm[7], left_hand[6], right_hand[6]
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
from PIL import Image


JOINT_POSITION_KEY = "puppet/joint_position"
END_EFFECTOR_KEY = "puppet/end_effector"
RGB_CAMERA_TOP_KEY = "observations/rgb_images/camera_top"
LANGUAGE_RAW_KEY = "language_raw"

SOURCE = "puppet"
ACTION_MODE = "absolute"
IMAGE_KEY = RGB_CAMERA_TOP_KEY
BENCHMARK_IMAGE_KEY_HINT = "camera_head"


class XsensSchemaError(ValueError):
    """Structured schema validation error for TienKung Xsens HDF5 files."""

    def __init__(
        self,
        *,
        path: str | Path,
        field: str,
        expected: str,
        actual: str,
        reason: str,
    ) -> None:
        self.path = str(path)
        self.field = field
        self.expected = expected
        self.actual = actual
        self.reason = reason
        super().__init__(
            f"Xsens schema error in {self.path}: field={field!r}, "
            f"expected={expected!r}, actual={actual!r}, reason={reason}"
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "reason": self.reason,
        }


@dataclass
class XsensTrajectory:
    path: Path
    task: str
    num_frames: int
    state: np.ndarray
    action: np.ndarray
    rgb_camera_top: Sequence[bytes] | Sequence[np.ndarray]
    metadata: dict[str, Any]


def _schema_error(path: str | Path, field: str, expected: str, actual: Any, reason: str) -> XsensSchemaError:
    return XsensSchemaError(
        path=path,
        field=field,
        expected=expected,
        actual=str(actual),
        reason=reason,
    )


def _require_dataset(file: h5py.File, key: str, path: str | Path) -> h5py.Dataset:
    if key not in file:
        raise _schema_error(path, key, "existing HDF5 dataset", "missing", "required field is absent")
    dataset = file[key]
    if not isinstance(dataset, h5py.Dataset):
        raise _schema_error(path, key, "HDF5 dataset", type(dataset).__name__, "field is not a dataset")
    return dataset


def _decode_scalar_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_scalar_text(value.item())
    return str(value)


def _read_task(file: h5py.File, path: str | Path) -> str:
    dataset = _require_dataset(file, LANGUAGE_RAW_KEY, path)
    if dataset.shape[0] < 1:
        raise _schema_error(path, LANGUAGE_RAW_KEY, "non-empty language_raw dataset", dataset.shape, "empty task dataset")

    task = _decode_scalar_text(dataset[0]).strip()
    if not task:
        raise _schema_error(path, LANGUAGE_RAW_KEY, "non-empty UTF-8 task string", repr(task), "task is empty")
    return task


def _validate_numeric_2d(
    dataset: h5py.Dataset,
    *,
    path: str | Path,
    key: str,
    width: int,
) -> None:
    if dataset.ndim != 2:
        raise _schema_error(path, key, f"2D numeric dataset with shape (T, {width})", dataset.shape, "wrong rank")
    if dataset.shape[0] <= 0:
        raise _schema_error(path, key, f"shape (T, {width}) with T > 0", dataset.shape, "empty time dimension")
    if dataset.shape[1] != width:
        raise _schema_error(path, key, f"shape (T, {width})", dataset.shape, "wrong feature dimension")
    if not np.issubdtype(dataset.dtype, np.number):
        raise _schema_error(path, key, "numeric dtype", dataset.dtype, "non-numeric dataset")


def _encoded_frame_to_bytes(frame: Any) -> bytes:
    if isinstance(frame, bytes):
        return frame
    if isinstance(frame, np.bytes_):
        return bytes(frame)
    array = np.asarray(frame)
    if array.dtype == np.uint8:
        return array.tobytes()
    return bytes(array)


def decode_rgb_frame(encoded_frame: Any) -> np.ndarray:
    """Decode one RoboMIND encoded RGB frame into an RGB uint8 array."""

    encoded = _encoded_frame_to_bytes(encoded_frame)
    with Image.open(io.BytesIO(encoded)) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.asarray(image)


def _validate_camera_top(dataset: h5py.Dataset, *, path: str | Path, expected_frames: int) -> tuple[int, int]:
    if dataset.ndim != 1:
        raise _schema_error(path, RGB_CAMERA_TOP_KEY, "1D encoded image sequence", dataset.shape, "wrong rank")
    if dataset.shape[0] != expected_frames:
        raise _schema_error(
            path,
            RGB_CAMERA_TOP_KEY,
            f"length {expected_frames}",
            dataset.shape,
            "camera frame count does not match state frame count",
        )

    try:
        image = decode_rgb_frame(dataset[0])
    except Exception as exc:  # noqa: BLE001 - keep dependency-specific image errors wrapped.
        raise _schema_error(
            path,
            RGB_CAMERA_TOP_KEY,
            "first frame decodable as RGB",
            type(exc).__name__,
            str(exc),
        ) from exc

    if image.ndim != 3 or image.shape[2] != 3:
        raise _schema_error(path, RGB_CAMERA_TOP_KEY, "RGB image with shape (H, W, 3)", image.shape, "decoded image is not RGB")
    height, width = image.shape[:2]
    return (width, height)


def _attrs_to_metadata(attrs: h5py.AttributeManager) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, np.generic):
            metadata[key] = value.item()
        elif isinstance(value, bytes):
            metadata[key] = value.decode("utf-8", errors="replace")
        else:
            metadata[key] = value
    return metadata


def build_canonical26(joint_position: np.ndarray, end_effector: np.ndarray) -> np.ndarray:
    """Build canonical XVLA 26D vector from TienKung Xsens arrays."""

    if joint_position.ndim != 2 or joint_position.shape[1] != 14:
        raise ValueError(f"joint_position must have shape (T, 14), got {joint_position.shape}")
    if end_effector.ndim != 2 or end_effector.shape[1] != 12:
        raise ValueError(f"end_effector must have shape (T, 12), got {end_effector.shape}")
    if joint_position.shape[0] != end_effector.shape[0]:
        raise ValueError(
            "joint_position and end_effector must have the same time dimension, "
            f"got {joint_position.shape[0]} and {end_effector.shape[0]}"
        )

    return np.concatenate(
        [
            joint_position[:, 0:7],
            joint_position[:, 7:14],
            end_effector[:, 0:6],
            end_effector[:, 6:12],
        ],
        axis=1,
    )


def read_xsens_trajectory(
    path: str | Path,
    *,
    decode_images: bool = False,
    dtype: np.dtype = np.float32,
) -> XsensTrajectory:
    """Read and validate one RoboMIND TienKung Xsens trajectory."""

    hdf5_path = Path(path).expanduser()
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as file:
        task = _read_task(file, hdf5_path)
        joint_dataset = _require_dataset(file, JOINT_POSITION_KEY, hdf5_path)
        effector_dataset = _require_dataset(file, END_EFFECTOR_KEY, hdf5_path)
        camera_dataset = _require_dataset(file, RGB_CAMERA_TOP_KEY, hdf5_path)

        _validate_numeric_2d(joint_dataset, path=hdf5_path, key=JOINT_POSITION_KEY, width=14)
        _validate_numeric_2d(effector_dataset, path=hdf5_path, key=END_EFFECTOR_KEY, width=12)
        if joint_dataset.shape[0] != effector_dataset.shape[0]:
            raise _schema_error(
                hdf5_path,
                END_EFFECTOR_KEY,
                f"time dimension {joint_dataset.shape[0]}",
                effector_dataset.shape,
                "end_effector frame count does not match joint_position",
            )

        num_frames = int(joint_dataset.shape[0])
        image_size = _validate_camera_top(camera_dataset, path=hdf5_path, expected_frames=num_frames)

        joint_position = joint_dataset[()].astype(dtype, copy=False)
        end_effector = effector_dataset[()].astype(dtype, copy=False)
        canonical26 = build_canonical26(joint_position, end_effector).astype(dtype, copy=False)
        if not np.isfinite(canonical26).all():
            raise _schema_error(
                hdf5_path,
                "canonical26",
                "finite numeric values",
                "contains NaN or inf",
                "canonical vector contains non-finite values",
            )

        if decode_images:
            rgb_frames: Sequence[np.ndarray] | Sequence[bytes] = [decode_rgb_frame(camera_dataset[index]) for index in range(num_frames)]
        else:
            rgb_frames = [_encoded_frame_to_bytes(camera_dataset[index]) for index in range(num_frames)]

        metadata = _attrs_to_metadata(file.attrs)
        metadata.update(
            {
                "path": str(hdf5_path),
                "source": SOURCE,
                "action_mode": ACTION_MODE,
                "language_raw": task,
                "has_language_distilbert": "language_distilbert" in file,
                "num_frames": num_frames,
                "rgb_camera_top_size": image_size,
                "image_key": IMAGE_KEY,
                "benchmark_image_key_hint": BENCHMARK_IMAGE_KEY_HINT,
            }
        )

    return XsensTrajectory(
        path=hdf5_path,
        task=task,
        num_frames=num_frames,
        state=canonical26.copy(),
        action=canonical26.copy(),
        rgb_camera_top=rgb_frames,
        metadata=metadata,
    )


def inspect_xsens_trajectory(path: str | Path) -> dict[str, Any]:
    """Return a compact inspection dictionary for one Xsens trajectory."""

    trajectory = read_xsens_trajectory(path, decode_images=False)
    first_frame_bytes = trajectory.rgb_camera_top[0]
    first_frame = decode_rgb_frame(first_frame_bytes)
    return {
        "path": str(trajectory.path),
        "task": trajectory.task,
        "num_frames": trajectory.num_frames,
        "state_shape": tuple(trajectory.state.shape),
        "action_shape": tuple(trajectory.action.shape),
        "state_dtype": str(trajectory.state.dtype),
        "action_dtype": str(trajectory.action.dtype),
        "state_min": float(np.min(trajectory.state)),
        "state_max": float(np.max(trajectory.state)),
        "contains_nan": bool(np.isnan(trajectory.state).any() or np.isnan(trajectory.action).any()),
        "first_rgb_shape": tuple(first_frame.shape),
        "first_rgb_dtype": str(first_frame.dtype),
        "metadata": trajectory.metadata,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a RoboMIND TienKung Xsens trajectory.hdf5 file.")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to trajectory.hdf5. Defaults to ROBOMIND_XSENS_SAMPLE.",
    )
    parser.add_argument("--decode-images", action="store_true", help="Decode all RGB frames to arrays during read.")
    args = parser.parse_args()

    path = args.path
    if path is None:
        import os

        path = os.environ.get("ROBOMIND_XSENS_SAMPLE")
    if not path:
        raise SystemExit("Provide a trajectory path or set ROBOMIND_XSENS_SAMPLE.")

    if args.decode_images:
        trajectory = read_xsens_trajectory(path, decode_images=True)
        result = inspect_xsens_trajectory(path)
        result["decoded_frame_count"] = len(trajectory.rgb_camera_top)
    else:
        result = inspect_xsens_trajectory(path)

    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

