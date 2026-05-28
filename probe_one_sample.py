#!/usr/bin/env python3
import h5py


PATH = (
    "/home/slzheng/datasets/RoboMIND2.0-Franka-Part-3/data/franka/"
    "place_coke_bottle_on_tray/success_episodes/0416_130751/data/trajectory.hdf5"
)


def scalar_to_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return repr(value)


with h5py.File(PATH, "r") as file:
    print("metadata")
    for key, value in file["metadata"].attrs.items():
        print(f"  {key}: {value!r}")

    print("\ncamera config")
    for group_name in ["camera_model", "camera_color_channel", "camera_color_resolution", "camera_depth_resolution"]:
        print(f"  {group_name}")
        for camera_name, dataset in file[group_name].items():
            value = dataset[()]
            if getattr(value, "shape", ()) == ():
                value = scalar_to_text(value)
            else:
                value = value.tolist()
            print(f"    {camera_name}: {value}")

    print("\nsample[0]")
    print("  camera_observations")
    camera_group = file["camera_observations"]
    print(f"    timestamp: {int(camera_group['timestamp'][0])}")
    print(f"    is_intervene: {bool(camera_group['is_intervene'][0])}")
    for image_kind in ["color_images", "depth_images"]:
        print(f"    {image_kind}")
        for camera_name, dataset in camera_group[image_kind].items():
            item = dataset[0]
            print(
                f"      {camera_name}: stored item type={type(item).__name__}, "
                f"shape={getattr(item, 'shape', None)}, dtype={getattr(item, 'dtype', None)}, bytes={len(item)}"
            )

    for root_name in ["master", "puppet"]:
        print(f"  {root_name}")
        for stream_name, stream_group in file[root_name].items():
            data = stream_group["data"][0]
            preview = data.tolist()
            print(
                f"    {stream_name}: data_shape_per_step={data.shape}, dtype={data.dtype}, "
                f"timestamp={int(stream_group['timestamp'][0])}, "
                f"is_intervene={bool(stream_group['is_intervene'][0])}, data={preview}"
            )
