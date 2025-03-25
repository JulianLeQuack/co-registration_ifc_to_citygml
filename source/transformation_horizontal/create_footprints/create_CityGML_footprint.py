import matplotlib.pyplot as plt
import numpy as np

import xml.etree.ElementTree as ET

from shapely.geometry import MultiPoint
from shapely.ops import substring


def create_CityGML_footprint(path_to_CityGML):

    points = []

    # Parse the CityGML file
    tree = ET.parse(path_to_CityGML)
    root = tree.getroot()

    # Define namespaces
    ns = {
        'bldg': 'http://www.opengis.net/citygml/building/2.0',
        'gml': 'http://www.opengis.net/gml'
    }

    # Find all ground surfaces in the CityGML file
    for gs in root.findall('.//bldg:GroundSurface', ns):
        # Locate the posList that contains the coordinates
        for posList in gs.findall('.//gml:posList', ns):
            # Assume coordinates are space separated
            coords = list(map(float, posList.text.split()))
            # Extract only x and y for the 2D convex hull
            for i in range(0, len(coords), 3):
                x = coords[i]
                y = coords[i+1]
                points.append((x, y))

    # Check if points were extracted
    if not points:
        raise ValueError("No ground surface points found in the CityGML file.")

    # Create a MultiPoint object and compute its convex hull
    ground_surface_points = MultiPoint(points)
    ground_surface_convex_hull = ground_surface_points.convex_hull.boundary

    # Sample many points along the boundary for alignment
    ground_surface_convex_hull_densified = MultiPoint()
    for i in np.arange(0, ground_surface_convex_hull.length, 0.2):
        s = substring(ground_surface_convex_hull, i, i+0.2)
        ground_surface_convex_hull_densified = ground_surface_convex_hull_densified.union(s.boundary)

    #Create np array from Multipoint object
    result = np.array([(point.x,point.y) for point in ground_surface_convex_hull_densified.geoms])

    return result

if __name__ == "__main__":
    ground_surface_convex_hull_densified = create_CityGML_footprint("./test_data/citygml/DEBY_LOD2_4959457.gml")
    plt.figure()
    x = [p.x for p in ground_surface_convex_hull_densified.geoms]
    y = [p.y for p in ground_surface_convex_hull_densified.geoms]
    plt.scatter(x, y)
    plt.grid(True)
    plt.show()
