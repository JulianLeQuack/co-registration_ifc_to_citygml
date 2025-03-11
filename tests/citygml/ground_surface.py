import xml.etree.ElementTree as ET
from shapely.geometry import MultiPoint, Polygon
from shapely import concave_hull
from shapely.ops import substring
import shapely.plotting
from ..matplotlib.Plotter import Plotter
import matplotlib.pyplot as plt
import numpy as np
import shapely

# Parse the CityGML file (update the file path as needed)
tree = ET.parse("./test_data/citygml/DEBY_LOD2_4959457.gml")
root = tree.getroot()

# Define namespaces (adjust if your file uses different prefixes)
ns = {
    'bldg': 'http://www.opengis.net/citygml/building/2.0',
    'gml': 'http://www.opengis.net/gml'
}

points = []

# Find all ground surfaces in the CityGML file
for gs in root.findall('.//bldg:GroundSurface', ns):
    # Locate the posList that contains the coordinates
    for posList in gs.findall('.//gml:posList', ns):
        # Assume coordinates are space separated; they might be in triplets (x, y, z)
        coords = list(map(float, posList.text.split()))
        # If coordinates are 3D, extract only x and y for the 2D convex hull
        # (adjust the step if using 2D only)
        for i in range(0, len(coords), 3):
            x = coords[i]
            y = coords[i+1]
            points.append((x, y))

# Check if points were extracted
if not points:
    raise ValueError("No ground surface points found in the CityGML file.")

# Create a MultiPoint object and compute its convex hull
multipoint = MultiPoint(points)
# polygon = Polygon(points)
ground_surface_convex_hull = multipoint.convex_hull
print(ground_surface_convex_hull)
ground_surface_convex_hull_outline = ground_surface_convex_hull.boundary
print(ground_surface_convex_hull_outline)
mp = MultiPoint()
for i in np.arange(0, ground_surface_convex_hull_outline.length, 0.2):
    s = substring(ground_surface_convex_hull_outline, i, i+0.2)
    mp = mp.union(s.boundary)
print(mp.geoms[0])
# ground_surface_concave_hull = concave_hull(geometry=multipoint, ratio=0.207)
# outer_line = polygon.exterior
# outer_line = unary_union(polygon)

# print("The convex hull of the ground surface is:")
# print(convex_hull)

# Plotter.plot_polygon(convex_hull)
# print(ground_surface_concave_hull)
# Plotter.plot_polygons(polygon, ground_surface_concave_hull)
# Plotter.plot_points(points)

plt.figure(figsize=(8, 8))
x = [p.x for p in mp.geoms]
y = [p.y for p in mp.geoms]
plt.scatter(x, y)
plt.grid(True)
plt.show()
