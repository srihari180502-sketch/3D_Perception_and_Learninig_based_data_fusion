# 3D Perception and Learning-Based Data Fusion

Course projects on 3D perception with the [KITTI object detection dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d): from LiDAR geometry and 3D bounding boxes, to camera–LiDAR fusion for object detection, to learning-based fusion with neural networks.

| Part | Topic | Folder |
|---|---|---|
| 1 | LiDAR point clouds and 3D bounding boxes | [`Part_1 Object detection & Estimation of 3D bounding box-submission`](Part_1%20Object%20detection%20%26%20Estimation%20of%203D%20bounding%20box-submission) |
| 2 | Camera–LiDAR fusion for object detection (classical ML) | [`Part_2 Camera-LiDAR fusion object detection-submission`](Part_2%20Camera-LiDAR%20fusion%20object%20detection-submission) |
| 3 | Learning-based fusion with neural networks | *coming next* |

---

## Part 1: LiDAR 3D Bounding Boxes

Processes KITTI LiDAR scans and visualises annotated objects with oriented 3D bounding boxes.

**Processing stages**

- Loading and inspecting LiDAR point clouds
- LiDAR-to-camera coordinate transformation with the calibration matrices
- Parsing KITTI object labels
- Building oriented 3D bounding boxes
- Multi-object 3D visualisation with in-box point counting
- Height-coloured point-cloud visualisation

**Required tools:** Python 3, NumPy, Matplotlib, Open3D, and KITTI sample data in `../data/kitti_mini/training/` with matching `velodyne/`, `label_2/` and `calib/` files.

**Run** (from the Part 1 folder):

```
python main.py
```

The script processes the frame IDs listed in `main.py`.

**Outputs:** height-coloured top-down point-cloud plots, an interactive Open3D viewer for the configured `viewer_frame`, and a console summary of each labelled object's forward distance and in-box point count.

---

## Part 2: Camera–LiDAR Fusion for Object Detection

Builds on Part 1 by predicting objects from raw sensor data and scoring the predictions against the KITTI labels. Three classical (non-deep-learning) detectors for Car, Van, Truck, Pedestrian and Cyclist are compared on the same 600 validation frames:

- **Camera-only:** HOG features and a linear SVM per class, slid over a multi-scale image pyramid, with hard-negative mining.
- **LiDAR-only:** RANSAC ground removal, DBSCAN clustering and a random forest on geometric cluster features, with learned 2D box refinement.
- **Fusion:** the LiDAR clusters, classified by a random forest that also receives camera evidence for each cluster box (feature-level fusion).

**Results** (2D IoU ≥ 0.5, KITTI moderate difficulty, average precision in %):

| Detector | Car | Van | Truck | Pedestrian | Cyclist | mAP |
|---|---|---|---|---|---|---|
| Camera-only | 8.9 | 0.8 | 0.4 | 0.4 | 0.7 | 2.2 |
| LiDAR-only | 87.1 | 56.2 | 42.0 | 45.9 | 72.9 | 60.8 |
| Fusion | 88.1 | 59.2 | 42.6 | 46.5 | 73.5 | 62.0 |

LiDAR dominates in this classical setting, and camera evidence adds a small precision gain. The Part 2 README explains the failure analysis (missing truck clusters, loose pedestrian boxes, the limits of single-template HOG), the role of partial ground truth, and the deviation from the official KITTI protocol.

**Required tools:** Python 3 with NumPy, Pillow, scikit-image, scikit-learn, joblib and Matplotlib (`pip install -r requirements.txt`), and the KITTI training split with `image_2/`, `label_2/`, `velodyne/` and `calib/`.

**Run** (from the Part 2 folder):

```
python kitti_fusion.py --data "path/to/KITTI_object_dataset/training"
```

Add `--quick` for a short test run. Results, plots and detection images are written to `kitti_fusion_output/`; the report is in the folder's `report/` directory.

---

## Part 3: Learning-Based Fusion (planned)

Replaces the hand-designed features of Part 2 with neural networks that learn from camera and LiDAR jointly, and predicts full 3D bounding boxes (position, size and orientation) evaluated against the 3D annotations. Part 2's results serve as the classical baseline.

## Dataset

A. Geiger, P. Lenz and R. Urtasun, "Are we ready for autonomous driving? The KITTI vision benchmark suite," CVPR, 2012. The dataset is not included in this repository; download it from the KITTI website.
