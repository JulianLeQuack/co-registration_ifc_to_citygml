import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt

def main():
    # Path to the input IFC file
    ifc_path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"
    ifc_file = ifcopenshell.open(ifc_path)
    
    # Configure the geometry settings.
    settings = ifcopenshell.geom.settings()
    
    # List to accumulate all polygon faces (projected to XY plane)
    polygons = []

    # Process each wall element (or change 'IfcWall' to any other IFC entity type)
    for element in ifc_file.by_type('IfcWall'):
        try:
            shape = ifcopenshell.geom.create_shape(settings, element)
        except Exception as e:
            print(f"Skipping element (ID: {element.id() if hasattr(element, 'id') else 'N/A'}) due to error: {e}")
            continue

        # Retrieve the list of vertices and faces (triangles)
        verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
        faces = ifcopenshell.util.shape.get_faces(shape.geometry)

        # Get the 4x4 transformation matrix as a numpy array.
        matrix = ifcopenshell.util.shape.get_shape_matrix(shape)

        # Apply the transformation to each vertex (local to global conversion)
        transformed_verts = []
        for v in verts:
            v_homog = np.array([v[0], v[1], v[2], 1.0])
            v_transformed = matrix @ v_homog
            transformed_verts.append(v_transformed[:3])  # Only use x, y, z

        # For each face (triangle), create a polygon (using the XY projection)
        for face in faces:
            # face is a list of vertex indices; retrieve the vertices.
            poly_vertices = [transformed_verts[i] for i in face]
            # Project the 3D coordinates to 2D by taking only the X and Y values.
            poly_xy = [(v[0], v[1]) for v in poly_vertices]
            polygons.append(poly_xy)

    if not polygons:
        print("No polygons extracted.")
        return

    # Convert each polygon coordinate list into a Shapely Polygon.
    shapely_polygons = []
    for coords in polygons:
        # Ensure there are at least 3 points to form a valid polygon.
        if len(coords) < 3:
            continue
        # Shapely polygons require the coordinate ring to be closed.
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            poly = Polygon(coords)
            if poly.is_valid:
                shapely_polygons.append(poly)
        except Exception as ex:
            print(f"Error creating polygon: {ex}")
    
    if shapely_polygons:
        # Create a MultiPolygon from the list of Shapely Polygons.
        multi_poly = MultiPolygon(shapely_polygons)

        # Plot the MultiPolygon using Matplotlib.
        fig, ax = plt.subplots()
        for poly in multi_poly.geoms:
            # Extract the exterior coordinates and plot the filled polygon.
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='lightblue', ec='blue')
            # Plot any interior rings (holes) if they exist.
            for interior in poly.interiors:
                ix, iy = interior.xy
                ax.fill(ix, iy, alpha=0.5, fc='white', ec='red')
        ax.set_title("MultiPolygon Extracted from IFC File")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal')
        plt.show()
    else:
        print("No valid polygons available to create a MultiPolygon.")

if __name__ == '__main__':
    main()
