import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape

import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union, polygonize


def is_wall_external(wall):
    """
    Attempt to find an 'IsExternal' property for the given wall.
    Returns True if found and True, otherwise False.
    """
    if not wall.IsDefinedBy:
        return False

    for rel_def in wall.IsDefinedBy:
        # We look for an IfcRelDefinesByProperties → IfcPropertySet → "IsExternal"
        if rel_def.is_a("IfcRelDefinesByProperties"):
            prop_def = rel_def.RelatingPropertyDefinition
            # Some IFC files store property sets in lists; handle both single vs. multiple
            prop_sets = prop_def if isinstance(prop_def, list) else [prop_def]
            
            for pset in prop_sets:
                if pset.is_a("IfcPropertySet"):
                    for prop in pset.HasProperties:
                        # We look for a property named "IsExternal"
                        if prop.Name == "IsExternal":
                            # This should be an IfcPropertySingleValue
                            try:
                                # If the property is True, it's an external wall
                                return bool(prop.NominalValue.wrappedValue)
                            except:
                                pass
    return False


def create_IFC_footprint(path_to_IFC):
    # 1. Open the IFC and get walls
    ifc_file = ifcopenshell.open(path_to_IFC)
    all_walls = ifc_file.by_type("IfcWallStandardCase")

    # 2. Filter for external walls
    external_walls = []
    for wall in all_walls:
        if is_wall_external(wall):
            external_walls.append(wall)

    if not external_walls:
        print("No external walls found or 'IsExternal' property not set.")
        return None

    # 3. Collect edges from these external walls
    lines = []
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)

    for wall in external_walls:
        try:
            shape = ifcopenshell.geom.create_shape(settings, wall)
            verts = shape.geometry.verts  # [x0, y0, z0, x1, y1, z1, ...]
            edges = ifcopenshell.util.shape.get_edges(shape.geometry)

            for edge in edges:
                coords_2d = []
                for idx in edge:
                    x = verts[idx * 3]
                    y = verts[idx * 3 + 1]
                    coords_2d.append((x, y))
                # Create a LineString for each edge
                lines.append(LineString(coords_2d))

        except Exception as e:
            print(f"Failed to process wall {wall.GlobalId}: {e}")

    if not lines:
        print("No edges extracted from external walls.")
        return None

    # 4. Merge/unify all external wall edges into a single geometry
    multiline = MultiLineString(lines)
    merged = unary_union(multiline)

    # 5. Polygonize the merged geometry
    polygons = list(polygonize(merged))
    if not polygons:
        print("No polygons could be formed from external walls.")
        return None

    # 6. Pick the largest polygon by area → building footprint
    largest_polygon = max(polygons, key=lambda p: p.area)

    # 7. Densify its exterior boundary
    exterior = largest_polygon.exterior
    distances = np.arange(0, exterior.length, 0.2)  # sample interval
    densified_coords = [exterior.interpolate(d).coords[0] for d in distances]

    return np.array(densified_coords)


if __name__ == "__main__":
    path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"
    footprint = create_IFC_footprint(path)
    if footprint is not None:
        plt.figure(figsize=(15, 10))
        plt.scatter(footprint[:, 0], footprint[:, 1], marker=".", color="blue")
        plt.title("Building Footprint (External Walls Only)")
        plt.grid(True)
        plt.show()
