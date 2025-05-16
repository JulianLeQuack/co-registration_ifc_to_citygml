import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import numpy as np

def find_IFC_extent(ifc_path):
    ifc_file = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    min_z, max_z = float("inf"), float("-inf")

    # List of main construction element types
    main_elements = ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcFooting"]

    for element_type in main_elements:
        for element in ifc_file.by_type(element_type):
            try:
                shape = ifcopenshell.geom.create_shape(settings, element)
            except Exception as e:
                print(f"Skipping element due to error: {e}")
                continue

            verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
            matrix = ifcopenshell.util.shape.get_shape_matrix(shape)

            for v in verts:
                v_homog = np.array([v[0], v[1], v[2], 1.0])
                v_transformed = matrix @ v_homog
                z = v_transformed[2]  # Extract the Z-coordinate
                min_z = min(min_z, z)
                max_z = max(max_z, z)

    if min_z == float("inf") or max_z == float("-inf"):
        print("No valid geometry found in the IFC file.")
        return None, None

    return round(min_z, 2), round(max_z, 2)

if __name__ == "__main__":
    ifc_path = "./test_data/ifc/3D_01_05_0501.ifc"
    min_z, max_z = find_IFC_extent(ifc_path)
    if min_z is not None and max_z is not None:
        print(f"Z-Extent of the IFC model (main construction elements): Min Z = {min_z}, Max Z = {max_z}")
    else:
        print("Failed to calculate Z-extent.")