import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image
import cv2

data_root = "C://Users//0109491s//PycharmProjects//dataset//KITTI//training"

image = Path(data_root) / "image_2/000120.png"
calib = Path(data_root) / "calib/000120.txt"
label_file = Path(data_root) / "label_2/000120.txt"

left_color_image = Image.open(image)


def draw_box_3d_v2(image, pixel_coords_norm, color=(255, 0, 0), thickness=2):
    """
    Draw a 3D bounding box projected in 2D on an image.

    Parameters:
    - image: The image array on which to draw.
    - pixel_coords_norm: The normalized pixel coordinates (2x8 array).
    - color: Color of the bounding box (default is red).
    - thickness: Line thickness for the bounding box.
    """
    # List of edges connecting the vertices of the bounding box in order
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Front face edges
        (4, 5), (5, 6), (6, 7), (7, 4),  # Back face edges
        (0, 4), (1, 5), (2, 6), (3, 7)  # Connecting edges between front and back faces
    ]

    # Iterate over the edges and draw lines between the vertices
    for start, end in edges:
        start_point = tuple(pixel_coords_norm[:, start].astype(int))
        end_point = tuple(pixel_coords_norm[:, end].astype(int))
        cv2.line(image, start_point, end_point, color, thickness)

    return image

class ObjectLabel3D:
    def __init__(self, line: str):
        """
        Kitti Label: [category truncation occlusion alpha x0 y0 x1 y1 h w l x y z yaw]
        """
        self.values = line.strip().split(" ")
        self.category = self.values[0]
        self.truncation = float(self.values[1])
        self.occlusion = float(self.values[2])
        self.alpha = float(self.values[3])
        # Pixel cordinate system
        self.x0 = float(self.values[4])
        self.y0 = float(self.values[5])
        self.x1 = float(self.values[6])
        self.y1 = float(self.values[7])
        # Camera cordinate system
        self.h, self.w, self.l = float(self.values[8]), float(self.values[9]), float(self.values[10])
        self.x, self.y, self.z = float(self.values[11]), float(self.values[12]), float(self.values[13])
        self.yaw = float(self.values[14])

    def get_2d_bbox(self, format="xyxy"):
        if format == "xyxy":
            return np.array([self.x0, self.y0, self.x1, self.y1])
        elif format == "xywh":
            return np.array([self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0])
        else:
            raise ValueError("Bbox format not supported.")

    def get_3d_bbox(self, coordinate="camera"):
        """
        Retrieve 3d bounding box as 8 corners in 3d space. See Figure-2 for
        for details on how the corners are labelled and how the rotation matrix is used.
        """
        # In camera coordinate the z-axis is the depth axis and y-axis is the height axis(pointing down).
        # Width is along x-axis
        if coordinate == "camera":
            # Rotate in x-z plane since y is the vertical axis for camera coordinate system.
            # Right handed coordinate system, rotate in counter-clockwise, x->y, y->z, z->x
            # and the angle is measured from x instead of z.
            rot_matrix = np.array(
                                 [
                                    [np.cos(self.yaw), 0, np.sin(self.yaw)],
                                    [0, 1, 0],
                                    [-np.sin(self.yaw), 0, np.cos(self.yaw)]
                                  ]
                                 )
            # coordinates of 8 corners of a bbox at origin(homogenous)
            box_corners_origin = np.array([
                                            [-self.l/2, -self.l/2, self.l/2, self.l/2, -self.l/2, -self.l/2, self.l/2, self.l/2],
                                            [     0,         0,        0,        0,     -self.h,   -self.h,   -self.h,  -self.h],
                                            [self.w/2, -self.w/2, -self.w/2, self.w/2, self.w/2, -self.w/2, -self.w/2, self.w/2],
                                            [     1,         1,        1,        1,        1,         1,         1,        1   ]
                                         ])
            transform = np.concatenate((rot_matrix, np.array([[self.x], [self.y], [self.z]])), axis=1)
            # Rotated box with center now translated to (x,y,z) from (0,0,0)
            box_corners = np.matmul(transform, box_corners_origin)
        elif coordinate == "velodyne":
            pass

        return box_corners

# Step: 1
calib_file = open(calib, 'r')
matrices_str = calib_file.readlines()
matrices = dict()

for line in matrices_str:
    if line.strip("\n").strip():
        name, matrix = line.strip("\n").split(": ")
        name = name.strip()
        matrix = np.array([float(x) for x in matrix.strip().split()], dtype=np.float32)
        if name.startswith("P"):
            matrix = matrix.reshape(3, 4)
        elif name.startswith("R"):
            matrix = matrix.reshape(3, 3)
        else:
            matrix = matrix.reshape(3, 4)
        matrices[name] = matrix

# Step: 2
labels = open(label_file, "r").read()
labels = labels.split("\n")
objects = []
for label in labels:
    if label.strip():
        objects.append(ObjectLabel3D(label))

rgb = np.array(left_color_image)

# Step: 3
for o in objects:
    rect_box = np.matmul(matrices["R0_rect"], o.get_3d_bbox())
    pixel_coords = np.matmul(matrices["P2"], np.concatenate((rect_box, np.ones((1, rect_box.shape[1]))), axis=0))
    pixel_coords_norm = pixel_coords[:2, :] / pixel_coords[2:, :]
    rgb = draw_box_3d_v2(rgb, pixel_coords_norm)

plt.imshow(rgb)
plt.show()