import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

def create_IFC_footprint_polygon(ifc_path, ifc_type="IfcSlab", tolerance=0.0):
    """
    Reads an IFC file and extracts geometry for elements of type `ifc_type`, then computes and returns:
    
      - multi_poly: A MultiPolygon with each individual extracted polygon.
      - footprint: A MultiPolygon that is the union of all valid polygon geometries (footprint).
      - simplified_polys: A list of simplified footprint polygons.
    
    Parameters:
        ifc_path (str): Path to the IFC file.
        ifc_type (str): IFC element type to process (default "IfcSlab").
        tolerance (float): Simplification tolerance (default 0.0, i.e. no simplification).
    
    Returns:
        tuple: (multi_poly, footprint, simplified_polys)
    """
    ifc_file = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    elements = ifc_file.by_type(ifc_type)
    polygons = []

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
            # Convert to homogeneous coordinates and apply transformation
            v_homog = np.array([v[0], v[1], v[2], 1.0])
            v_transformed = matrix @ v_homog
            transformed_verts.append(v_transformed[:3])

        for face in faces:
            poly_vertices = [transformed_verts[i] for i in face]
            poly_xy = [(v[0], v[1]) for v in poly_vertices]
            polygons.append(poly_xy)

    if not polygons:
        print("No polygons extracted.")
        return None, None, None

    # Build a list of valid shapely Polygons
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
        print("No valid polygons available to create a MultiPolygon.")
        return None, None, None

    multi_poly = MultiPolygon(shapely_polygons)

    # Compute the union of all polygons; this combines overlapping areas while
    # preserving disjoint areas. The result might be a Polygon or MultiPolygon.
    union_poly = unary_union(shapely_polygons)
    if union_poly.geom_type == "Polygon":
        footprint = MultiPolygon([union_poly])
    elif union_poly.geom_type == "MultiPolygon":
        footprint = union_poly
    else:
        print("Unexpected geometry type:", union_poly.geom_type)
        return None, None, None

    # Simplify each polygon in the footprint using the provided tolerance.
    simplified_polys = [poly.simplify(tolerance=tolerance, preserve_topology=True) for poly in footprint.geoms]

    return multi_poly, footprint, simplified_polys

def main():
    # Choose your IFC file path:
    # ifc_path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"
    ifc_path = "./test_data/ifc/3.003 01-05-0507_EG.ifc"

    multi_poly, footprint, simplified_polys = create_IFC_footprint_polygon(ifc_path, ifc_type="IfcSlab", tolerance=0.0)
    
    if multi_poly is None or footprint is None or simplified_polys is None:
        print("Failed to create IFC footprint polygon.")
        return

    # Set up plots to visualize:
    # 1. The full slab geometry (the original MultiPolygon).
    # 2. The simplified footprint outlines.
    # 3. The vertices of the simplified footprint polygons.
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Subplot 1: Full slab geometry.
    for poly in multi_poly.geoms:
        x, y = poly.exterior.xy
        ax1.fill(x, y, alpha=0.5, fc='lightblue', ec='blue')
        for interior in poly.interiors:
            ix, iy = interior.xy
            ax1.fill(ix, iy, alpha=0.5, fc='white', ec='red')
    ax1.set_title("Extracted Slab MultiPolygon")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect('equal')

    # Subplot 2: Simplified footprint outlines.
    for poly in simplified_polys:
        outline_x, outline_y = poly.exterior.xy
        ax2.plot(outline_x, outline_y, color="green", linewidth=2)
    ax2.set_title("Simplified Slab Footprint Outlines")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect('equal')

    # Subplot 3: Vertices of simplified footprints.
    for poly in simplified_polys:
        x, y = poly.exterior.xy
        ax3.scatter(x, y, color="green")
    ax3.set_title("Vertices of Simplified Footprints")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
