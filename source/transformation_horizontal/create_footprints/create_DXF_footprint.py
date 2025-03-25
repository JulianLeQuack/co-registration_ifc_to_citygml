import matplotlib.pyplot as plt
import numpy as np

import ezdxf

from shapely.geometry import MultiPoint
from shapely.ops import substring


def create_DXF_footprint(path_to_DXF):

    points = []

    # Parse DXF file and create Modelspace
    file = ezdxf.readfile(path_to_DXF)
    msp = file.modelspace()

    # Query CAD Objects in layer "A_01_TRAGWAND"
    wall_layer_name = "A_01_TRAGWAND"
    wall_entities = msp.query('*[layer=="A_01_TRAGWAND"]')

    # Iterate over all entites in layer
    for entity in wall_entities:
        # Extract virtual entites from ACAD_PROXY_ENTITy
        object = entity.virtual_entities()
        # Extract entities from virtual entities
        for geometry in object:
            # Extract POLYLINE entities
            if geometry.dxftype() == "POLYLINE":
                # Ectract Vertices list from Polylines
                vertices = geometry.points()
                # Extract Vertices from Vertices List
                for vertex in vertices:
                    points.append(vertex)

    # Check if points were extracted
    if not points:
        raise ValueError("No ground surface points found in the DXF file.")

    footprint_points = MultiPoint(points)
    footprint_convex_hull = footprint_points.convex_hull.boundary

    # Sample many points along the boundary for alignment
    footprint_convex_hull_densified = MultiPoint()
    for i in np.arange(0, footprint_convex_hull.length, 0.2):
        s = substring(footprint_convex_hull, i, i+0.2)
        footprint_convex_hull_densified = footprint_convex_hull_densified.union(s.boundary)

    return footprint_convex_hull_densified

if __name__ == "__main__":
    footprint_convex_hull_densified = create_DXF_footprint("./test_data/dxf/01-05-0501_EG.dxf")
    plt.figure()
    x = [p.x for p in footprint_convex_hull_densified.geoms]
    y = [p.y for p in footprint_convex_hull_densified.geoms]
    plt.scatter(x, y)
    plt.grid(True)
    plt.show()