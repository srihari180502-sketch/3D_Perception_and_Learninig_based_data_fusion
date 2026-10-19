
# Libraries used for working with point-cloud data, plotting, and 3D viewing.
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

project_name = "Object detection with LiDAR pointclouds & 3D bounding box estimation"


frame_id= "000016"  # Example frame setting; the batch at the bottom chooses the frames processed.
viewer_frame = "000001"  # Open the interactive 3D viewer for this frame only.


# Path helps build file locations that work independently of the terminal's current folder.
from pathlib import Path

# Build the 8 corners and 12 edges of one KITTI 3D label box.
# KITTI fields 8-10 are height, width, and length; fields 11-13 are the
# box's bottom-center location; field 14 is its rotation around camera Y.
def make_box(object_fields, color):
    height = float(object_fields[8])
    width = float(object_fields[9])
    length = float(object_fields[10])
    center_x = float(object_fields[11])
    bottom_y = float(object_fields[12])
    center_z = float(object_fields[13])
    yaw = float(object_fields[14])

    # Start with four rectangle corners in the box's local (unrotated) X-Z plane.
    half_length = length / 2
    half_width = width / 2

    local_x = np.array([half_length, half_length, -half_length, -half_length])
    local_z = np.array([half_width, -half_width, -half_width, half_width])

    # Rotate the rectangle by its yaw angle, then move it to its camera-frame location.
    box_x = np.cos(yaw) * local_x + np.sin(yaw) * local_z + center_x
    box_z = -np.sin(yaw) * local_x + np.cos(yaw) * local_z + center_z

    # Camera Y points down, so subtracting the height gives the top of the box.
    bottom_corners = np.column_stack(
        (box_x, np.full(4, bottom_y), box_z)
    )
    top_corners = np.column_stack(
        (box_x, np.full(4, bottom_y - height), box_z)
    )
    corners = np.vstack((bottom_corners, top_corners))

    # Each pair gives the corner indices connected by one line of the cuboid.
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    # Open3D LineSet displays the corners and connecting edges as a wireframe box.
    box = o3d.geometry.LineSet()
    box.points = o3d.utility.Vector3dVector(corners)
    box.lines = o3d.utility.Vector2iVector(edges)
    box.colors = o3d.utility.Vector3dVector(
        [color for edge in edges]
    )

    return box

# Return a True/False mask identifying which camera-frame points lie inside a label box.
def points_inside_box(points, object_fields):
    height = float(object_fields[8])
    width = float(object_fields[9])
    length = float(object_fields[10])
    center_x = float(object_fields[11])
    bottom_y = float(object_fields[12])
    center_z = float(object_fields[13])
    yaw = float(object_fields[14])

    # Shift points so the box center is the origin, then undo its yaw rotation.
    dx = points[:, 0] - center_x
    dy = points[:, 1] - bottom_y
    dz = points[:, 2] - center_z

    local_x = np.cos(yaw) * dx - np.sin(yaw) * dz
    local_z = np.sin(yaw) * dx + np.cos(yaw) * dz

    # Keep points within the box's length, width, and vertical height limits.
    return (
        (np.abs(local_x) <= length / 2)
        & (np.abs(local_z) <= width / 2)
        & (dy >= -height)
        & (dy <= 0)
    )

# Load one KITTI frame, transform its points into camera coordinates, and inspect its labels.
def process_frame(frame_id):
    print("KITTI frame:", frame_id)
    
    # The script is inside Part1; the dataset is stored beside it in the project data folder.
    data_folder = Path(__file__).resolve().parent.parent / "data" / "kitti_mini" / "training"  

    # Each .bin file stores one LiDAR scan; each .txt file stores labels or calibration values.
    scan_path = data_folder / "velodyne" / f"{frame_id}.bin"
    #print("Current folder:", Path.cwd( ))
    #print("Checking path:", scan_path.resolve())
    print("Scan file found:", scan_path.exists())

    #source ~/.venvs/lidar-3d-object-detection/bin/activate

    # Read the binary scan as 32-bit numbers. Four numbers make each point: x, y, z, reflectance.
    raw_values = np.fromfile(scan_path, dtype=np.float32)
    print("Number of stored values:", len(raw_values))

    # Inspect the first four stored values to confirm how a point is represented.

    first_point = raw_values[:4]
    print("First point (x, y, z, reflectance)", first_point)

    # Group every four values into one LiDAR point: x, y, z, and reflectance.
    lidar_points= raw_values.reshape(-1,4)
    print("Point cloud shape:", lidar_points.shape)
    print("First three points:")
    print(lidar_points[:3])

    # Keep only spatial coordinates for geometry operations; reflectance is not needed here.

    xyz_points = lidar_points[:, :3]
    print("xyz shape:", xyz_points.shape)
    print("First three xyz points:")
    print(xyz_points[:3])

    # Print the coordinate ranges as a quick check of the scan's spatial coverage.

    print("X min/max:", xyz_points[:,0].min(),xyz_points[:,0].max())
    print("Y min/max:", xyz_points[:,1].min(),xyz_points[:,1].max())
    print("Z min/max:", xyz_points[:,2].min(),xyz_points[:,2].max())

    # Make a top-down plot: horizontal axes are LiDAR x/y and point color represents height z.
    # The PNG is saved in the directory from which this script is run.

    plt.figure(figsize=(8,8))
    #plt.scatter(xyz_points[:,0],xyz_points[:,1], s=0.2)
    scatter = plt.scatter(xyz_points[:,0],xyz_points[:,1], c=xyz_points[:,2], s=0.2, cmap="turbo")
    plt.xlabel("x (meters)")
    plt.ylabel("y (meters)")
    plt.axis("equal")
    plt.colorbar(scatter,label="Height z (meters)")
    plt.savefig(f"{frame_id}_point_cloud_height.png", dpi=200)


    # Read this frame's KITTI ground-truth labels. These labels provide object classes and boxes;
    # this script visualizes those supplied labels rather than predicting objects by itself.

    label_path = data_folder / "label_2" / f"{frame_id}.txt"

    with label_path.open() as label_file:
        label_lines= label_file.readlines()

    print("label lines:", len(label_lines))
    print("First label:", label_lines[0])

    # A label line is whitespace-separated; split it into class, image data, box, and pose fields.

    fields = label_lines[0].split()

    # Example of converting the first label's dimensions from text into numeric values.

    height = float(fields[8])
    width = float(fields[9])
    length = float(fields[10])
    #print("Box dimensions in meters:", height, width, length)


    # Keep the object categories this project currently handles; DontCare and other types are skipped.
    supported_classes = ["Car", "Truck", "Pedestrian", "Cyclist", "Van"]
    objects_in_frame = []

    # Collect every supported object's fields so boxes and point counts can be handled in a loop.
    for line in label_lines:
        fields = line.split()

        if fields[0] in supported_classes:
            objects_in_frame.append(fields)
            print(fields[0])

    print("Number of supported objects:", len(objects_in_frame))

    # Also print all original label classes, including categories not used above.
    for line in label_lines:
        fields= line.split()
        print(fields[0])

    # Calibration tells us how to express LiDAR points in the camera coordinate system,
    # which is the coordinate system used by the KITTI 3D box labels.

    calib_path = data_folder / "calib" / f"{frame_id}.txt"
    #print(calib_path.read_text())

    # Find Tr_velo_to_cam, the LiDAR-to-camera transform, in the calibration text file.

    calibration_lines = calib_path.read_text().splitlines()

    for line in calibration_lines:
        if line.startswith("Tr_velo_to_cam:"):
            numbers_text= line.split(":",1)[1]
            matrix_values = np.fromstring(numbers_text,sep=" ")
            print("Number of matrix values:", len(matrix_values))

            # The 12 values form a 3x4 matrix: a 3x3 rotation followed by a translation column.

            transform_matrix = matrix_values.reshape(3,4)
            print("LiDAR-to-camera matrix:")
            print(transform_matrix)

    # Apply the rotation and translation to the first point as a small example.

    rotation = transform_matrix[:, :3]
    translation = transform_matrix[:, 3]

    first_lidar_point = xyz_points[0]
    first_camera_point = rotation @ first_lidar_point + translation

    print("First LiDAR point:", first_lidar_point)
    print("Same point in camera coordinates:", first_camera_point)

    # Find R0_rect, which aligns camera coordinates to KITTI's rectified camera frame.

    for line in calibration_lines:
        if line.startswith("R0_rect:"):
            values= np.fromstring(line.split(":",1)[1], sep=" ")
            rectification_matrix= values.reshape(3,3)

    first_rectified_point = rectification_matrix @ first_camera_point
    print("Point in rectified camera coordinates:", first_rectified_point)

    # Transform the full scan in one operation, then rectify every camera-frame point.
    camera_points = xyz_points @ rotation.T + translation 
    rectified_points = camera_points @ rectification_matrix.T

    print("Rectified point cloud shape:", rectified_points.shape)
    print("First rectified point:", rectified_points[0])

    # Open the interactive 3D viewer only for the selected viewer_frame.
    # Points start gray; points inside each labeled box are colored by object class.
    if frame_id == viewer_frame:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(rectified_points)
    
            # RGB colors used consistently for each supported object's points and box.
            class_colors = {
                "Car": [0.0, 0.4, 1.0],
                "Truck": [0.0, 1.0, 0.0],
                "Pedestrian": [1.0, 0.0, 0.0],
                "Cyclist": [1.0, 0.0, 1.0],
                "Van": [1.0, 0.8, 0.0],
            }
    
            # Give every point a neutral gray, then recolor returns inside each object box.
            colors = np.full((len(rectified_points), 3), 0.65)
    
            for object_fields in objects_in_frame:
                class_name = object_fields[0]
                inside_object = points_inside_box(rectified_points, object_fields)
                colors[inside_object] = class_colors[class_name]
    
            cloud.colors = o3d.utility.Vector3dVector(colors)
    
            # Add coordinate axes, then add a wireframe cuboid for every supported label.
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    
            geometries = [cloud, axes]
    
            for object_fields in objects_in_frame:
                class_name = object_fields[0]
                box = make_box(object_fields, class_colors[class_name])
                geometries.append(box)
    
            # Display the point cloud, coordinate axes, and all object boxes together.
            o3d.visualization.draw_geometries(geometries)

    # Count scan returns inside each labeled box and save a summary for this frame.
    # This is a geometric point count, not a learned detector confidence score.
    object_results = []

    for object_fields in objects_in_frame:
        object_class = object_fields[0]
        object_x = float(object_fields[11])
        object_y = float(object_fields[12])
        object_z = float(object_fields[13])

        # Reuse the box mask to count how many LiDAR points fall inside this object's cuboid.
        inside_object = points_inside_box(rectified_points, object_fields)
        point_count = int(np.count_nonzero(inside_object))

        print(
            f"{object_class} location: "
            f"{object_x:.2f}, {object_y:.2f}, {object_z:.2f}"
        )
        print(f"LiDAR points inside {object_class} box: {point_count}")

        object_results.append(
            (frame_id, object_class, object_z, point_count)
        )

    if not objects_in_frame:
        print("No supported labeled objects in this frame.")

    # Close the saved plot so figures do not accumulate while processing many frames.
    plt.close("all")
    return object_results

# Process these selected sample frames. Add or remove frame IDs to change the batch.
frame_ids = [
    "000001", "000004", "000007", "000008", "000009",
    "000010", "000011", "000012", "000015", "000016",
    "000019", "000021", "000023", "000025", "000031",
    "000032", "000043", "000048", "000049", "000061",
]
all_object_results = []

# Run the same processing steps for each frame and combine their per-object results.
for frame_id in frame_ids:
    all_object_results.extend(process_frame(frame_id))

# Print each object's class, camera Z (forward depth), and number of scan points in its box.
print("\nObject-box point summary:")
for frame_id, object_class, distance, point_count in all_object_results:
    print(
        f"{frame_id}: {object_class}, "
        f"forward distance = {distance:.2f} m, "
        f"points = {point_count}"
        
    )
