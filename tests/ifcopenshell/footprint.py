import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape

from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint
from shapely.ops import substring

import matplotlib.pyplot as plt
import numpy as np

ifc_file_path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"

ifc_file = ifcopenshell.open(ifc_file_path)

wall_standard_cases = ifc_file.by_type("IfcWallStandardCase")
walls = ifc_file.by_type("IfcWall")

shapes = []

for wall in wall_standard_cases:
    try:
        element = wall
        settings = ifcopenshell.geom.settings()
        shape = ifcopenshell.geom.create_shape(settings, element)
        shapes.append(shape)
    except:
        pass

# for wall in walls:
#     try:
#         element = wall
#         settings = ifcopenshell.geom.settings()
#         shape = ifcopenshell.geom.create_shape(settings, element)
#         shapes.append(shape)
#     except:
#         pass

# print(len(wall_standard_cases))
# print(len(walls))
# print(len(shapes))

points_2d = []

for shape in shapes:
    vertices = shape.geometry.verts
    for i in range(0, len(vertices), 3):
        x = vertices[i]
        y = vertices[i+1]
        points_2d.append((x, y))

footprint_points = MultiPoint(points_2d)
footprint_convex_hull = footprint_points.convex_hull.boundary

print(footprint_points)
print(footprint_convex_hull)

footprint_convex_hull_densified = MultiPoint()
for i in np.arange(0, footprint_convex_hull.length, 0.2):
    s = substring(footprint_convex_hull, i, i+0.2)
    footprint_convex_hull_densified = footprint_convex_hull_densified.union(s.boundary)

plt.figure()
x_original = [p[0] for p in points_2d]
y_original = [p[1] for p in points_2d]
x = [p.x for p in footprint_convex_hull_densified.geoms]
y = [p.y for p in footprint_convex_hull_densified.geoms]
plt.scatter(x, y, label="Densified Convex Hull", color="blue")
plt.scatter(x_original, y_original, label="Original Points", color="red")
plt.legend()
plt.grid(True)
plt.show()

# ## Tests with one wall element

# wall = ifc_file.by_type("IfcWallStandardCase")[1]
# settings = ifcopenshell.geom.settings()
# shape = ifcopenshell.geom.create_shape(settings, wall)

# vertices = shape.geometry.verts
# points_2d = []
# for i in range(0, len(vertices), 3):
#     x = vertices[i] * 1000
#     y = vertices[i+1] * 1000
#     points_2d.append((x, y))

# plt.figure()
# x_original = [p[0] for p in points_2d]
# y_original = [p[1] for p in points_2d]
# plt.scatter(x_original, y_original, label="Original Points")
# plt.grid(True)
# plt.show()
