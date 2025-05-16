import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import numpy as np

def find_IFC_extent(ifc_path):
    ifc_file = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    main_elements = ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcFooting"]

    storey_extents = []

    # Get all storeys in the model
    storeys = ifc_file.by_type("IfcBuildingStorey")
    for storey in storeys:
        min_z, max_z = float("inf"), float("-inf")
        # Get all elements related to this storey
        related_elements = []
        for rel in getattr(storey, "ContainsElements", []):
            related_elements.extend(rel.RelatedElements)
        # Filter only main construction elements
        elements = [el for el in related_elements if el.is_a() in main_elements]
        for element in elements:
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
                z = v_transformed[2]
                min_z = min(min_z, z)
                max_z = max(max_z, z)
        if min_z != float("inf") and max_z != float("-inf"):
            storey_extents.append({
                "storey_id": storey.GlobalId,
                "storey_name": getattr(storey, "Name", ""),
                "min_z": round(min_z, 2),
                "max_z": round(max_z, 2)
            })
        else:
            storey_extents.append({
                "storey_id": storey.GlobalId,
                "storey_name": getattr(storey, "Name", ""),
                "min_z": None,
                "max_z": None
            })
    return storey_extents

if __name__ == "__main__":
    ifc_path = "./test_data/ifc/3D_01_05_0501.ifc"
    storey_extents = find_IFC_extent(ifc_path)
    for ext in storey_extents:
        print(f"Storey '{ext['storey_name']}' (ID: {ext['storey_id']}): Min Z = {ext['min_z']}, Max Z = {ext['max_z']}")