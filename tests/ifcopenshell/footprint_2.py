import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import numpy as np
import matplotlib.pyplot as plt

ifc_file = ifcopenshell.open("./test_data/ifc/3.002 01-05-0501_EG.ifc")

wall_standard_cases = ifc_file.by_type("IfcWallStandardCase")
walls = ifc_file.by_type("IfcWall")

shapes = []

for wall in wall_standard_cases:
    try:
        settings = ifcopenshell.geom.settings()
        shape = ifcopenshell.geom.create_shape(settings, wall)
        matrix = ifcopenshell.util.shape.get_shape_matrix(shape)
        grouped_verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
        grouped_verts_global = []
        for vert in grouped_verts:
            local_homo = np.append(vert, 1)
            global_homo = np.dot(local_homo, matrix)
        shapes.append(global_homo[0:3])
    except Exception as e:
        print(f"Error processing wall {wall.GlobalId}: {str(e)}")
        raise

print(shapes[0])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x = [p[0] for p in shapes]
y = [p[1] for p in shapes]
z = [p[2] for p in shapes]

ax.set_xlabel('X Axis Label')
ax.set_ylabel('Y Axis Label')
ax.set_zlabel('Z Axis Label')

ax.scatter(x,y,z)
plt.grid(True)
plt.show()

points_2d = []

for shape in shapes:
    x = shape[0]
    y = shape[2]
    points_2d.append((x,y))

print(points_2d[0])