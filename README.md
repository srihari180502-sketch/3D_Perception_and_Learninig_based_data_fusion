# Part 1: LiDAR Point Cloud and 3D Bounding Boxes

## About this project

This project uses LiDAR scans from the [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/). It transforms the points into camera coordinates and displays KITTI's labelled objects with 3D bounding boxes.


## Requirements

- Python 3
- NumPy
- Matplotlib
- Open3D

Install the libraries in the same Python environment you will use to run the program:

```bash
python3 -m pip install numpy matplotlib open3d
```

On Windows, use `python -m pip install numpy matplotlib open3d` if your Python command is `python` instead of `python3`.

## KITTI data folder

The data is not included in this Part 1 folder. Place the KITTI files in the project-level `data/kitti_mini/training` folder. The folders and files should look like this:

```text
lidar-3d-object-detection/
├── data/
│   └── kitti_mini/
│       └── training/
│           ├── velodyne/  # LiDAR scan files, for example 000001.bin
│           ├── label_2/   # Object labels, for example 000001.txt
│           └── calib/     # Calibration files, for example 000001.txt
└── Part1_Object detection with estimation of 3D Bounding boxes in LiDAR pointclouds/
    ├── main.py
    └── README.md
```

For every frame used by the program, the scan, label, and calibration files must have the same six-digit frame ID. For example, frame `000001` needs `000001.bin` in `velodyne` and `000001.txt` in both `label_2` and `calib`.

## Run the program

1. Open a terminal in this Part 1 folder.
2. Activate the Python environment where NumPy, Matplotlib, and Open3D are installed.
3. Run:

   ```bash
   python3 main.py
   ```

   On Windows, use `python main.py` if that is the command for your Python installation.

## What the program does

- Reads the selected KITTI LiDAR frames and prints basic point-cloud information.
- Uses calibration matrices to transform LiDAR points into rectified camera coordinates.
- Reads the labelled classes, dimensions, locations, and orientations for cars, trucks, pedestrians, cyclists, and vans.
- Draws a 3D box for each supported labelled object and highlights LiDAR points inside the boxes.
- Opens the interactive Open3D viewer for the frame selected by `viewer_frame` in `main.py`.
- Saves a top-down point-cloud height plot for each processed frame in the directory where the program is run.
- Prints each labelled object's forward distance and the number of LiDAR points inside its box.

The point count is a geometric count of LiDAR returns inside a labelled box. It is not a detector confidence score or a measure of detection accuracy.

To change which frames are processed, edit the `frame_ids` list near the bottom of `main.py`. To choose which one opens in the 3D viewer, change `viewer_frame` near the top. Close the Open3D window to let the program continue processing the remaining frames.
