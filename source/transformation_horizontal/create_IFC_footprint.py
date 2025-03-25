import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape

import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import MultiPoint
from shapely.ops import substring

def create_IFC_footprint(path_to_IFC):

    # Initialize list for vertices
    points = []

    # Import IFC file
    ifc_file = ifcopenshell.open(path_to_IFC)

    # Extract IfcWallStandardCase elements from the Ifc File. They will be used to create the building footprint.
    walls = ifc_file.by_type("IfcWallStandardCase")

    # Use world coordinates, otherwise all wall elements are in a local CRS around 0,0
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)

    # Extract vertices from wall elements and remove z-value
    for wall in walls:
        try:
            shape = ifcopenshell.geom.create_shape(settings, wall)

            grouped_verts = ifcopenshell.util.shape.get_vertices(shape.geometry)

            # Remove z-value so we get a 2D footprint
            xy_points = grouped_verts[:, :2]
            points.extend(xy_points.tolist())
        except Exception as e:
            print(f"Failed to process wall {wall.GlobalId}: {e}")

    # Check, if points list is populated
    if not points:
        print("No footprint points extracted from the IFC file.")
        return

    # Create shapely multipoint for further processing
    footprint_points = MultiPoint(points)
    # Create convex hull around footprint points
    footprint_convex_hull = footprint_points.convex_hull.boundary

    # Sample points along the convex hull for transformation estimation
    footprint_convex_hull_densified = MultiPoint()
    for i in np.arange(0, footprint_convex_hull.length, 0.2):
        s = substring(footprint_convex_hull, i, i+0.2)
        footprint_convex_hull_densified = footprint_convex_hull_densified.union(s.boundary)

    return footprint_convex_hull_densified
    

if __name__ == "__main__":
    footprint_convex_hull_densified = create_IFC_footprint("./test_data/ifc/3.002 01-05-0501_EG.ifc")
    plt.figure()
    x = [p.x for p in footprint_convex_hull_densified.geoms]
    y = [p.y for p in footprint_convex_hull_densified.geoms]
    plt.scatter(x, y)
    plt.grid(True)
    plt.show()  