# RoboMind HDF5 Tools

Utilities for inspecting and converting RoboMIND Franka trajectory data.

This project was built around the RoboMIND Franka Part 3 dataset downloaded under:

```text
/home/slzheng/datasets/RoboMIND2.0-Franka-Part-3
```

The included scripts focus on one representative trajectory by default:

```text
/home/slzheng/datasets/RoboMIND2.0-Franka-Part-3/data/franka/place_coke_bottle_on_tray/success_episodes/0416_130751/data/trajectory.hdf5
```

## Environment

The scripts were tested in the local conda environment named `RoboMind`.

```bash
conda activate RoboMind
```

Required packages:

```bash
pip install modelscope h5py numpy
```

For PyTorch conversion:

```bash
pip install torch
```

If you use decoded image tensors instead of encoded image bytes:

```bash
pip install pillow
```

## Scripts

### Inspect HDF5 Structure

Prints the full HDF5 group/dataset tree, including shape, dtype, compression, chunks, and attributes.

```bash
python inspect_hdf5_structure.py
```

Inspect another HDF5 file:

```bash
python inspect_hdf5_structure.py /path/to/trajectory.hdf5
```

### Probe One Sample

Prints the metadata, camera configuration, and `sample[0]` contents for the default trajectory.

```bash
python probe_one_sample.py
```

### Convert HDF5 To PyTorch

Converts a RoboMIND `trajectory.hdf5` file into a PyTorch `.pt` file.

```bash
python convert_hdf5_to_torch.py
```

By default, images are stored as encoded `uint8` tensors containing JPEG/PNG bytes. This avoids eagerly expanding every image into dense pixel tensors.

Include a timestep-wise `samples` list:

```bash
python convert_hdf5_to_torch.py --include-samples
```

Decode images into tensors:

```bash
python convert_hdf5_to_torch.py --decode-images
```

Use custom input/output paths:

```bash
python convert_hdf5_to_torch.py \
  --input /path/to/trajectory.hdf5 \
  --output /path/to/trajectory.pt
```

## Documentation

See `trajectory_hdf5_structure.md` for a Chinese tree-form explanation of the example HDF5 file, including field meanings and a concrete `sample[0]`.

## Data Notes

Large dataset files are intentionally not tracked by git. Keep raw HDF5 files and generated PyTorch `.pt` files outside the repository, or under ignored data/output directories.
