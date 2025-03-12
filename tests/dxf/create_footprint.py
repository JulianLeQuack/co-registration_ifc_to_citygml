import ezdxf
import ezdxf.entities
from shapely.geometry import MultiPoint
from shapely.ops import substring
import matplotlib.pyplot as plt
import numpy as np

file = ezdxf.readfile("./test_data/dxf/01-05-0501_EG.dxf")

msp = file.modelspace()

wall_layer_name = "A_01_TRAGWAND"
wall_entities = msp.query('*[layer=="A_01_TRAGWAND"]')

points = []

for entity in wall_entities:
    # print(points)
    # print("entity\n", entity)
    object = entity.virtual_entities()
    # print("object\n", object)
    for geometry in object:
        if geometry.dxftype() == "POLYLINE":
            # print("geometry\n", geometry)
            vertices = geometry.points()
            # print("vertices\n", vertices)
            for vertex in vertices:
                # point = vertex.dxf.location
                points.append(vertex)

print(points[:10])

footprint_points = MultiPoint(points)
footprint_convex_hull = footprint_points.convex_hull.boundary

footprint_convex_hull_densified = MultiPoint()
for i in np.arange(0, footprint_convex_hull.length, 0.2):
    s = substring(footprint_convex_hull, i, i+0.2)
    footprint_convex_hull_densified = footprint_convex_hull_densified.union(s.boundary)


plt.figure(figsize=(8, 8))
x_origin = [p.x for p in points]
y_origin = [p.y for p in points]
x_hull = [p.x for p in footprint_convex_hull_densified.geoms]
y_hull = [p.y for p in footprint_convex_hull_densified.geoms]
plt.scatter(x_origin, y_origin)
plt.scatter(x_hull, y_hull)
plt.grid(True)
plt.show()

# ## Rudimental Tests

# walls = msp.query('*[layer=="A_01_TRAGWAND"]')

# if not walls:
#     print("No walls.")

# acad_proxy_entity = walls[0]
# generator_object = acad_proxy_entity.virtual_entities()
# poly_line = next(generator_object)
# poly_line_2 = next(generator_object)
# poly_face_mesh = poly_line.get_mode()
# vertices = poly_line.vertices
# vertex = vertices[0].dxf.location
# for vertex in vertices:
#     print(vertex.dxf.location)

# print(acad_proxy_entity)
# print(generator_object)
# print(poly_line)
# print(poly_line_2)
# print(poly_face_mesh)
# print(vertices)
# print(vertex)

# print(ezdxf.entities.polyline)

# # for wall in walls:
# #     wall = wall.explode(msp)

# # for wall in walls:
# #     print(wall)

# # for entity in msp:
# #     print(entity.dxf.layer)

# walls = msp.query('*[layer=="A_01_TRAGWAND"]')

# if not walls:
#     print("No walls.")

# # print(walls[0].load_proxy_graphic())


