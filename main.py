import numpy as np
import laspy
import open3d as o3d
import matplotlib.pyplot as plt

import graphic_utils
from boundary_box import BoundaryBox
from octree import Octree

#### Constants

# max number of possible subdivisions
subdivisions = 3

# default center point for the octotree 
# is average of min / max coordinates
center_offset = [0, 0, 0]

# display in console how long it takes to process
# points when building tree
debug = True

# remove points that are inside subdivision cube 
# but outside of same cube's sphere. 
# setting this to False leaves these points in the parent node
remove_points_outside_sphere = True

# processing the whole dataset is quite slow,
# so skipping points is possible for testing purposes
# set to 1 for full dataset
# set to n to only take every nth point
step = 10

# subdivision cube / sphere relative point density.
# lower value = more density
# in open3d screen, tweak visual density with '-' and '=' buttons if necessary
# recommended values: 100 - 100000
cube_density = 100
sphere_density = 200


#### Read dataset
point_data = []
with laspy.open('2743_1234.las') as fh:
    print('Points from Header:', fh.header.point_count)
    las = fh.read()
    point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))[::step]


#### Find center point and cube length for root
minX = min(point_data[:, 0])
maxX = max(point_data[:, 0])
minY = min(point_data[:, 1])
maxY = max(point_data[:, 1])
minZ = min(point_data[:, 2])
maxZ = max(point_data[:, 2])
center_point = [((minX + maxX) / 2) + center_offset[0], ((minY + maxY) / 2) + center_offset[1], ((minZ + maxZ) / 2) + center_offset[2]]
x_length = maxX - minX
y_length = maxY - minY
z_length = maxZ - minZ
cube_dim_half = max([x_length, y_length, z_length]) / 2

#### Create and build tree
tree = Octree(BoundaryBox(center_point, cube_dim_half), point_data, layers_left=subdivisions)
print("Building octree.")
tree.Build(debug=debug, remove_points_outside_sphere=remove_points_outside_sphere)


#### Test find / add
## tested with: subdivisions: 3, step: 10, no offset

## 0. print some of the points in the generated tree
# graphic_utils.print_points(tree)

# 1. finding points that should exist
print("-")
print(tree.find(np.array([29333,  89531, 219108])))
print("-")
# 2. finding points that shouldn't currently exist
print(tree.find(np.array([29363,  89531, 219108])))
print(tree.find(np.array([29363,  89531, 500000])))
print("-")
# 3. adding points. First point is valid and should be added, second point is invalid (z coord is too high to fit in any cube / sphere)
print(tree.add(np.array([29363,  89531, 219108])))
print(tree.add(np.array([29363,  89531, 500000])))
print("-")
# 4. Repeating #2. Only first point should exist
print(tree.find(np.array([29363,  89531, 219108])))
print(tree.find(np.array([29363,  89531, 500000])))
print("-")


#### Display generated tree
print("Displaying.")
geometries = []
graphic_utils.create_geometry(tree, cube_density, sphere_density, geometries=geometries)
o3d.visualization.draw_geometries(geometries)