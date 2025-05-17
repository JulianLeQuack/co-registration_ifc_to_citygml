import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import numpy as np


def extract_ifc_centroid(ifc_path):
    """
    Extracts the centroid of the IFC model.

    Args:
        ifc_path (str): Path to the IFC file.

    Returns:
        dict: A dictionary containing the centroid coordinates (x, y, z).
    """
    ifc_file = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    main_elements = ["IfcWall", "IfcSlab"]

    vertices = []

    # Get all elements in the model
    elements = ifc_file.by_type("IfcBuildingElement")
    for element in elements:
        if element.is_a() not in main_elements:
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, element)
        except Exception as e:
            continue
        verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
        matrix = ifcopenshell.util.shape.get_shape_matrix(shape)
        for v in verts:
            v_homog = np.array([v[0], v[1], v[2], 1.0])
            v_transformed = matrix @ v_homog
            vertices.append(v_transformed[:3])  # Only x, y, z

    if not vertices:
        return None

    centroid = np.mean(vertices, axis=0)
    return {"x": centroid[0], "y": centroid[1], "z": centroid[2]}

if __name__ == "__main__":
    ifc_path = "./test_data/ifc/3D_01_05_0501_transformed.ifc"
    centroid = extract_ifc_centroid(ifc_path)
    print(f"Centroid: {centroid}")
