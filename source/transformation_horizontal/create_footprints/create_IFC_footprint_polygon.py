import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

def create_IFC_footprint_polygon(ifc_path, ifc_type="IfcSlab"):
    """
    Reads an IFC file and extracts geometry for elements of type `ifc_type`,
    then computes and returns a footprint as a MultiPolygon that contains all
    disjoint polygon exteriors.
    
    Parameters:
        ifc_path (str): Path to the IFC file.
        ifc_type (str): IFC element type to process (default "IfcSlab").
    
    Returns:
        MultiPolygon: The union of all valid polygon geometries.
    """
    ifc_file = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    elements = ifc_file.by_type(ifc_type)
    polygons = []

    # Extract geometry for each element
    for element in elements:
        try:
            shape = ifcopenshell.geom.create_shape(settings, element)
        except Exception as e:
            print(f"Skipping element due to error: {e}")
            continue

        verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
        faces = ifcopenshell.util.shape.get_faces(shape.geometry)
        matrix = ifcopenshell.util.shape.get_shape_matrix(shape)

        transformed_verts = []
        for v in verts:
            v_homog = np.array([v[0], v[1], v[2], 1.0])
            v_transformed = matrix @ v_homog
            transformed_verts.append(v_transformed[:3])

        for face in faces:
            poly_vertices = [transformed_verts[i] for i in face]
            poly_xy = [(v[0], v[1]) for v in poly_vertices]
            polygons.append(poly_xy)

    if not polygons:
        print("No polygons extracted.")
        return None

    # Build a list of valid shapely Polygons.
    shapely_polygons = []
    for coords in polygons:
        if len(coords) < 3:
            continue
        # Ensure the polygon is closed (first point equals last point)
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            poly = Polygon(coords)
            if poly.is_valid:
                shapely_polygons.append(poly)
        except Exception as ex:
            print(f"Error creating polygon: {ex}")

    if not shapely_polygons:
        print("No valid polygons available to create a footprint.")
        return None

    # Compute the union of all valid polygons. This combines overlapping areas
    # while keeping disjoint parts separate.
    union_poly = unary_union(shapely_polygons)
    if union_poly.geom_type == "Polygon":
        footprint = MultiPolygon([union_poly])
    elif union_poly.geom_type == "MultiPolygon":
        footprint = union_poly
    else:
        print("Unexpected geometry type:", union_poly.geom_type)
        return None

    return footprint

def main():
    # Choose your IFC file path:
    # ifc_path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"
    ifc_path = "./test_data/ifc/3.003 01-05-0507_EG.ifc"

    footprint = create_IFC_footprint_polygon(ifc_path, ifc_type="IfcSlab")
    if footprint is None:
        print("Failed to create IFC footprint polygon.")
        return

    # Plot the footprint MultiPolygon
    plt.figure(figsize=(8, 8))
    for poly in footprint.geoms:
        x, y = poly.exterior.xy
        plt.plot(x, y, color="green", linewidth=2)
    plt.title("IFC Footprint MultiPolygon")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == '__main__':
    main()
