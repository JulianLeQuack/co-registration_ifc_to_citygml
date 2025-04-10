import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

def create_IFC_footprint(ifc_path, tolerance=5, ):
    """
    Extracts and simplifies the footprint from an IFC file using world coordinates.

    Parameters:
        ifc_path (str): Path to the IFC file.

    Returns:
        tuple: (multi_poly, footprint_simplified)
            - multi_poly: A MultiPolygon representing the collection of extracted polygons.
            - footprint_simplified: The simplified union (largest area) of the extracted polygons.
        Returns (None, None) if no valid polygons are extracted.
    """
    ifc_file = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    # Enable world coordinates to avoid manual transformation
    settings.set("use_world_coords", True)

    # Get wall elements from the IFC file
    elements = ifc_file.by_type("IfcWall")
    polygons = []

    for element in elements:
        try:
            shape = ifcopenshell.geom.create_shape(settings, element)
        except Exception as e:
            print(f"Skipping element due to error: {e}")
            continue

        # Get vertices and face definitions; vertices are already in world coordinates.
        verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
        faces = ifcopenshell.util.shape.get_faces(shape.geometry)

        # Build 2D polygon coordinates (using X and Y) for each face.
        for face in faces:
            poly_vertices = [verts[i] for i in face]
            poly_xy = [(v[0], v[1]) for v in poly_vertices]
            polygons.append(poly_xy)

    if not polygons:
        print("No polygons extracted.")
        return None, None

    shapely_polygons = []
    for coords in polygons:
        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            poly = Polygon(coords)
            if poly.is_valid:
                shapely_polygons.append(poly)
        except Exception as ex:
            print(f"Error creating polygon: {ex}")

    if not shapely_polygons:
        print("No valid polygons available to create a MultiPolygon.")
        return None, None

    multi_poly = MultiPolygon(shapely_polygons)
    union_poly = unary_union(shapely_polygons)

    if union_poly.geom_type == "Polygon":
        footprint = union_poly
    elif union_poly.geom_type == "MultiPolygon":
        footprint = max(union_poly.geoms, key=lambda p: p.area)
    else:
        print("Unexpected geometry type:", union_poly.geom_type)
        return None, None

    # Simplify the footprint boundary
    footprint_simplified = footprint.simplify(tolerance=0, preserve_topology=True)

    return multi_poly, footprint_simplified

def main():
    # Specify the IFC file path.
    ifc_path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"
    # ifc_path = "./test_data/ifc/3.003 01-05-0507_EG.ifc"
    
    multi_poly, footprint_simplified = create_IFC_footprint(ifc_path)

    if multi_poly is None or footprint_simplified is None:
        print("No footprint to display.")
        return

    # Plotting the extracted polygons and the simplified footprint.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the collection of original polygons.
    for poly in multi_poly.geoms:
        x, y = poly.exterior.xy
        ax1.fill(x, y, alpha=0.5, fc='lightblue', ec='blue')
        # Plot interiors (holes) if any.
        for interior in poly.interiors:
            ix, iy = interior.xy
            ax1.fill(ix, iy, alpha=0.5, fc='white', ec='red')
    ax1.set_title("Extracted Slab MultiPolygon")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect('equal')

    # Plot the simplified footprint outline.
    outline_x, outline_y = footprint_simplified.exterior.xy
    ax2.plot(outline_x, outline_y, color="green", linewidth=2)
    ax2.set_title("Simplified Slab Footprint Outline")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
