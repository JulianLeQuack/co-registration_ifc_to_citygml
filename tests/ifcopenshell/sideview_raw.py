import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt

def create_IFC_sideview_raw(ifc_path):
    ifc_file = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    polygons = []
    for element in ifc_file.by_type("IfcWall"):
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
            transformed_verts.append(v_transformed[1:])

        for face in faces:
            poly_vertices = [transformed_verts[i] for i in face]
            poly_xy = [(v[0], v[1]) for v in poly_vertices]
            polygons.append(poly_xy)

    if not polygons:
        print("No polygons extracted.")
        return None

    # Build a list of valid shapely Polygons (no union, no exteriors only)
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
        print("No valid polygons available.")
        return None

    return shapely_polygons

if __name__ == "__main__":
    from source.transformation_vertical.create_sideviews.create_CityGML_sideview import create_CityGML_sideview

    ifc_path = "./test_data/ifc/3D_01_05_0501_transformed_vertical.ifc"

    citygml_path = "./test_data/citygml/DEBY_LOD2_4959457.gml"
    citygml_sideview = create_CityGML_sideview(citygml_path=citygml_path, building_ids=["DEBY_LOD2_4959457"])

    polygons = create_IFC_sideview_raw(ifc_path)

    # Plot all raw polygons
    plt.figure(figsize=(8, 8))

    if not citygml_sideview.is_empty:
        for poly in citygml_sideview.geoms:
            y, z = poly.exterior.xy
            plt.plot(y, z, color='blue')
    
    # for poly in polygons:
    #     x, y = poly.exterior.xy
    #     plt.plot(x, y, color="green", linewidth=1)
    #     for interior in poly.interiors:
    #         ix, iy = interior.xy
    #         plt.plot(ix, iy, color="red", linewidth=1)
    plt.title("IFC Sideview Raw Polygons")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()