#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py


DEFAULT_HDF5 = (
    "/home/slzheng/datasets/RoboMIND2.0-Franka-Part-3/data/franka/"
    "place_coke_bottle_on_tray/success_episodes/0416_130751/data/trajectory.hdf5"
)


def format_attrs(h5_object):
    if not h5_object.attrs:
        return ""

    pieces = []
    for key, value in h5_object.attrs.items():
        pieces.append(f"{key}={repr(value)}")
    return " attrs: " + ", ".join(pieces)


def describe_node(name, h5_object):
    depth = 0 if not name else name.count("/") + 1
    indent = "  " * depth
    label = "/" if not name else Path(name).name

    if isinstance(h5_object, h5py.Dataset):
        compression = f", compression={h5_object.compression}" if h5_object.compression else ""
        chunks = f", chunks={h5_object.chunks}" if h5_object.chunks else ""
        print(
            f"{indent}- {label}: Dataset "
            f"shape={h5_object.shape}, dtype={h5_object.dtype}{compression}{chunks}"
            f"{format_attrs(h5_object)}"
        )
    elif isinstance(h5_object, h5py.Group):
        print(f"{indent}- {label}/: Group{format_attrs(h5_object)}")


def inspect_hdf5(path):
    hdf5_path = Path(path).expanduser()
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    print(f"File: {hdf5_path}")
    print(f"Size: {hdf5_path.stat().st_size / (1024 ** 2):.2f} MiB")

    with h5py.File(hdf5_path, "r") as file:
        describe_node("", file)
        file.visititems(describe_node)


def main():
    parser = argparse.ArgumentParser(description="Print the structure of a RoboMIND HDF5 file.")
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_HDF5,
        help=f"HDF5 file path. Defaults to: {DEFAULT_HDF5}",
    )
    args = parser.parse_args()
    inspect_hdf5(args.path)


if __name__ == "__main__":
    main()
