import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape

import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import LineString, Polygon, MultiLineString
from shapely.ops import substring, unary_union


def create_IFC_footprint(path_to_IFC) -> np.array:
    """
    Create a 2D footprint of the building from an IFC file.
    The footprint is created by extracting the vertices of the walls
    and removing the z-value. The footprint is returned as a numpy array.
    :param path_to_IFC: Path to the IFC file
    :return: Numpy array of footprint points
    """

    # Initialize list for vertices
    points = []  # Use a Python list to collect the points

    # Import IFC file
    ifc_file = ifcopenshell.open(path_to_IFC)

    # Extract IfcWallStandardCase elements from the Ifc File. They will be used to create the building footprint.
    walls = ifc_file.by_type("IfcWall")

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
            points.extend(xy_points)  # Extend the list with the new points
        except Exception as e:
            print(f"Failed to process wall {wall.GlobalId}: {e}")

    # Check, if points list is populated
    if not points:
        print("No footprint points extracted from the IFC file.")
        return np.array([]) #return empty array

    return np.array(points) #Return the points as np array


if __name__ == "__main__":
    # footprint = create_IFC_footprint("./test_data/ifc/3.002 01-05-0501_EG.ifc")
    footprint = create_IFC_footprint("./test_data/ifc/3.003 01-05-0507_EG.ifc")
    print(f"Output Type: {type(footprint)}")
    plt.figure(figsize=(15, 10))
    if footprint.size > 0:
        x = footprint[:, 0]
        y = footprint[:, 1]
        plt.scatter(x, y)
        plt.grid(True)
        plt.show()