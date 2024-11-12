import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pathlib import Path

# Set up paths
data_root = "C://Users//0109491s//PycharmProjects//dataset//KITTI//testing"
image_path = Path(data_root) / "image_2//000000.png"
lidar_path = Path(data_root) / "velodyne//000000.bin"
calib_path = Path(data_root) / "calib//000000.txt"

# Load calibration matrix
def load_calibration(calib_file):
    matrices = {}
    with open(calib_file, 'r') as f:
        for line in f.readlines():
            if "P2:" in line:
                matrices["P2"] = np.array(line.split()[1:], dtype=np.float32).reshape(3, 4)
            elif "R0_rect:" in line:
                matrices["R0_rect"] = np.array(line.split()[1:], dtype=np.float32).reshape(3, 3)
            elif "Tr_velo_to_cam:" in line:
                matrices["Tr_velo_to_cam"] = np.array(line.split()[1:], dtype=np.float32).reshape(3, 4)
    return matrices

# Load calibration matrices
matrices = load_calibration(calib_path)

# Read image
left_color_image = Image.open(image_path)

# Read LIDAR points and split into xyz coordinates and intensity
points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
lidar_xyz, intensity = points[:, :3], points[:, 3]

# Calculate norm
dist = np.sqrt(np.sum(np.square(lidar_xyz), axis=1))

# Convert LIDAR points to homogeneous coordinates
lidar_xyz_homogeneous = np.concatenate((lidar_xyz, np.ones((lidar_xyz.shape[0], 1))), axis=1)

# Project into camera coordinates
cam_xyz = np.matmul(matrices["Tr_velo_to_cam"], lidar_xyz_homogeneous.T)

# Transform to rectified coordinates
cam_rect_xyz = np.matmul(matrices["R0_rect"], cam_xyz)

# Convert camera coordinates to homogeneous coordinates
cam_rect_xyz_homogeneous = np.concatenate((cam_rect_xyz, np.ones((1, cam_rect_xyz.shape[1]))), axis=0)

# Project to image plane of camera 2 (left color camera)
left_color_image_su_sv_s = np.matmul(matrices["P2"], cam_rect_xyz_homogeneous)

# Filter points
left_color_image_su_sv_s = left_color_image_su_sv_s.T
valid_indices = left_color_image_su_sv_s[:, 2] > 0
left_color_image_su_sv_s = left_color_image_su_sv_s[valid_indices]

# Remove scale and convert to pixels
left_color_u_v = (left_color_image_su_sv_s[:, :2] / left_color_image_su_sv_s[:, 2:]).T

# Filter distance values for valid points only
dist = dist[valid_indices]

# Convert distance to color (you can customize this function)
def dist_to_color(dist):
    norm_dist = (dist - dist.min()) / (dist.max() - dist.min())
    color = (norm_dist * 255).astype(np.uint8)
    return color

color = dist_to_color(dist)

# Paint the image
hsv_image = cv2.cvtColor(np.array(left_color_image), cv2.COLOR_RGB2HSV)
for i in range(left_color_u_v.shape[1]):
    cv2.circle(hsv_image, (int(left_color_u_v[0][i]), int(left_color_u_v[1][i])), 2, (int(color[i]), 255, 255), -1)

# Convert back to RGB and display
rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
plt.imshow(rgb_image)
plt.show()
