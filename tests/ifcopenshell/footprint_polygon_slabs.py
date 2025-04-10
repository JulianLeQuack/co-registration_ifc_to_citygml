import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

def main():
    # ifc_path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"
    ifc_path = "./test_data/ifc/3.003 01-05-0507_EG.ifc"

    ifc_file = ifcopenshell.open(ifc_path)

    settings = ifcopenshell.geom.settings()
    elements = ifc_file.by_type("IfcSlab")

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
            v_homog = np.array([v[0], v[1], v[2], 1.0])
            v_transformed = matrix @ v_homog
            transformed_verts.append(v_transformed[:3])

        for face in faces:
            poly_vertices = [transformed_verts[i] for i in face]
            poly_xy = [(v[0], v[1]) for v in poly_vertices]
            polygons.append(poly_xy)

    if not polygons:
        print("No polygons extracted.")
        return

    # Build a list of valid shapely Polygons
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

    if shapely_polygons:
        multi_poly = MultiPolygon(shapely_polygons)

        # Compute union: this will combine all overlapping areas but keep disjoint parts separate.
        union_poly = unary_union(shapely_polygons)
        # If union returns a single Polygon, wrap it in a MultiPolygon.
        if union_poly.geom_type == "Polygon":
            footprint = MultiPolygon([union_poly])
        elif union_poly.geom_type == "MultiPolygon":
            footprint = union_poly
        else:
            print("Unexpected geometry type:", union_poly.geom_type)
            return

        # Simplify boundaries for each polygon in the footprint.
        # (Using a tolerance of 0 here preserves all vertices; adjust if needed for your data.)
        simplified_polys = [poly.simplify(tolerance=0, preserve_topology=True) for poly in footprint.geoms]

        # Set up plots to visualize:
        # 1. The full slab geometry from the original MultiPolygon.
        # 2. The simplified footprint outlines.
        # 3. The vertices of the simplified footprints.
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Subplot 1: full slab geometry.
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

        # Subplot 2: simplified footprint outlines.
        for poly in simplified_polys:
            outline_x, outline_y = poly.exterior.xy
            ax2.plot(outline_x, outline_y, color="green", linewidth=2)
        ax2.set_title("Simplified Slab Footprint Outlines")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_aspect('equal')

        # Subplot 3: vertices of the simplified footprints.
        for poly in simplified_polys:
            x, y = poly.exterior.xy
            ax3.scatter(x, y, color="green")
        ax3.set_title("Vertices of Simplified Footprints")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_aspect('equal')

        plt.tight_layout()
        plt.show()
    else:
        print("No valid polygons available to create a MultiPolygon.")

if __name__ == '__main__':
    main()
