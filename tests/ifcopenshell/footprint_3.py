import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint

def main():
    # Load the IFC file
    ifc_file = ifcopenshell.open('./test_data/ifc/3.002 01-05-0501_EG.ifc')
    
    # Retrieve all walls of type IfcWallStandardCase
    walls = ifc_file.by_type('IfcWallStandardCase')
    
    # List to store all footprint points (x, y)
    footprint_points = []
    
    # Configure geometry settings; setting USE_WORLD_COORDS to True ensures the transformation
    # matrix is applied so the vertices are in global coordinates.
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    
    # Process each wall element
    for wall in walls:
        try:
            shape = ifcopenshell.geom.create_shape(settings, wall)
            # Get the flattened list of vertex coordinates (x, y, z, x, y, z, ...)
            verts = shape.geometry.verts
            # Convert the list to a numpy array and reshape into an Nx3 array
            verts_np = np.array(verts, dtype=np.float32).reshape(-1, 3)
            # Extract the X and Y coordinates to form a 2D footprint
            xy_points = verts_np[:, :2]
            footprint_points.extend(xy_points.tolist())
        except Exception as e:
            print(f"Failed to process wall {wall.GlobalId}: {e}")
    
    # Ensure we have collected some points
    if not footprint_points:
        print("No footprint points extracted from the IFC file.")
        return

    # Convert the collected footprint points to a numpy array for processing
    footprint_points_np = np.array(footprint_points)
    
    # Create a MultiPoint object and compute its convex hull using Shapely
    multipoint = MultiPoint(footprint_points_np)
    convex_hull = multipoint.convex_hull
    print("Convex Hull:", convex_hull)
    
    # Plot the footprint points and the convex hull
    plt.figure(figsize=(8, 8))
    plt.scatter(footprint_points_np[:, 0], footprint_points_np[:, 1],
                s=1, color='blue', label='Wall Vertices')
    if convex_hull.geom_type == 'Polygon':
        x, y = convex_hull.exterior.xy
        plt.plot(x, y, color='red', linewidth=2, label='Convex Hull')
    plt.title("Convex Hull of IFC Walls Footprint")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()
