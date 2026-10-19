# Part 1: LiDAR 3D Bounding Boxes

This project processes KITTI LiDAR scans and visualizes annotated objects with oriented 3D bounding boxes.

## Process Done

- LiDAR point-cloud scanning and inspection
- LiDAR-to-camera coordinate transformation
- KITTI object-label parsing
- Construction of oriented 3D bounding boxes
- Multi-object 3D visualization and in-box point counting
- Point-cloud height visualization

## Required Tools

- Python 3
- NumPy, Matplotlib, and Open3D
- KITTI sample data at `../data/kitti_mini/training/`, with matching files in `velodyne/`, `label_2/`, and `calib/`

Run `python main.py` from this folder. The script processes the frame IDs listed in `main.py`.

## Outputs

- Height-colored top-down point-cloud plots, saved in the working directory.
- An interactive Open3D viewer for the configured `viewer_frame`.
- A console summary of each labelled object's forward distance and in-box point count.

