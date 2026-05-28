#!/usr/bin/env python3
import argparse
import io
from pathlib import Path

import h5py
import numpy as np
import torch


DEFAULT_INPUT = (
    "/home/slzheng/datasets/RoboMIND2.0-Franka-Part-3/data/franka/"
    "place_coke_bottle_on_tray/success_episodes/0416_130751/data/trajectory.hdf5"
)

CAMERA_NAMES = (
    "camera_front",
    "camera_left",
    "camera_right",
    "camera_top",
    "camera_wrist_left",
    "camera_wrist_right",
)

ROBOT_ROOTS = ("master", "puppet")


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_attrs(group):
    return {key: decode_scalar(value) for key, value in group.attrs.items()}


def read_scalar_dataset(dataset):
    return decode_scalar(dataset[()])


def as_tensor(array):
    return torch.from_numpy(np.asarray(array))


def encoded_image_tensor(dataset, index):
    return torch.from_numpy(np.asarray(dataset[index], dtype=np.uint8).copy())


def decoded_image_tensor(dataset, index, mode):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Decoded images require Pillow. Install it with: pip install pillow") from exc

    encoded = np.asarray(dataset[index], dtype=np.uint8)
    with Image.open(io.BytesIO(encoded.tobytes())) as image:
        if mode is not None:
            image = image.convert(mode)
        array = np.asarray(image)
    return torch.from_numpy(array.copy())


def read_camera_config(file):
    config = {}
    for group_name in (
        "camera_model",
        "camera_color_channel",
        "camera_color_resolution",
        "camera_depth_resolution",
    ):
        config[group_name] = {}
        for camera_name, dataset in file[group_name].items():
            value = read_scalar_dataset(dataset)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            config[group_name][camera_name] = value
    return config


def read_robot_arrays(file, root_name):
    root = file[root_name]
    output = {}
    for stream_name, stream_group in root.items():
        output[stream_name] = {
            "data": as_tensor(stream_group["data"][()]),
            "timestamp": as_tensor(stream_group["timestamp"][()]),
            "is_intervene": as_tensor(stream_group["is_intervene"][()]),
        }
    return output


def read_images(file, image_kind, decode_images):
    image_group = file["camera_observations"][image_kind]
    output = {}

    for camera_name in CAMERA_NAMES:
        dataset = image_group[camera_name]
        frames = []
        for index in range(dataset.shape[0]):
            if decode_images:
                mode = "RGB" if image_kind == "color_images" else None
                frames.append(decoded_image_tensor(dataset, index, mode=mode))
            else:
                frames.append(encoded_image_tensor(dataset, index))
        output[camera_name] = frames

    return output


def build_samples(payload):
    trajectory_length = int(payload["metadata"]["trajectory_length"])
    samples = []

    for index in range(trajectory_length):
        sample = {
            "index": index,
            "camera_observations": {
                "timestamp": payload["camera_observations"]["timestamp"][index],
                "is_intervene": payload["camera_observations"]["is_intervene"][index],
                "color_images": {
                    camera_name: payload["camera_observations"]["color_images"][camera_name][index]
                    for camera_name in CAMERA_NAMES
                },
                "depth_images": {
                    camera_name: payload["camera_observations"]["depth_images"][camera_name][index]
                    for camera_name in CAMERA_NAMES
                },
            },
            "master": {},
            "puppet": {},
        }

        for root_name in ROBOT_ROOTS:
            for stream_name, stream in payload[root_name].items():
                sample[root_name][stream_name] = {
                    "data": stream["data"][index],
                    "timestamp": stream["timestamp"][index],
                    "is_intervene": stream["is_intervene"][index],
                }

        samples.append(sample)

    return samples


def convert_one_hdf5(input_path, output_path, decode_images=False, include_samples=False):
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as file:
        payload = {
            "source_hdf5": str(input_path),
            "metadata": read_attrs(file["metadata"]),
            "camera_config": read_camera_config(file),
            "camera_observations": {
                "timestamp": as_tensor(file["camera_observations"]["timestamp"][()]),
                "is_intervene": as_tensor(file["camera_observations"]["is_intervene"][()]),
                "color_images": read_images(file, "color_images", decode_images=decode_images),
                "depth_images": read_images(file, "depth_images", decode_images=decode_images),
            },
            "master": read_robot_arrays(file, "master"),
            "puppet": read_robot_arrays(file, "puppet"),
        }

    if include_samples:
        payload["samples"] = build_samples(payload)

    torch.save(payload, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a RoboMIND trajectory.hdf5 file into a PyTorch .pt file."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input trajectory.hdf5 path. Defaults to: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        default="/home/slzheng/CY/RoboMind/trajectory_0416_130751.pt",
        help="Output .pt path.",
    )
    parser.add_argument(
        "--decode-images",
        action="store_true",
        help="Decode JPEG/PNG bytes into image tensors. Requires Pillow and uses much more disk/RAM.",
    )
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="Also store a samples list where samples[i] is the fully assembled timestep i.",
    )
    args = parser.parse_args()

    output_path = convert_one_hdf5(
        input_path=args.input,
        output_path=args.output,
        decode_images=args.decode_images,
        include_samples=args.include_samples,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
