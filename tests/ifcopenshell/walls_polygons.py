import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape

import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import LineString, MultiLineString, Polygon, MultiPoint
from shapely.ops import unary_union, polygonize, substring


def create_walls_footprints(path_to_ifc, sampling_interval=0.2):
    """
    For each IfcWallStandardCase in the IFC file, create a 2D polygon outline 
    and densify it using substring-based approach.
    Returns a dict: {wall_GlobalId: numpy_array_of_xy_points, ...}.
    """
    # 1. Open the IFC and retrieve walls
    ifc_file = ifcopenshell.open(path_to_ifc)
    walls = ifc_file.by_type("IfcWallStandardCase")
    
    # Set up geometry settings (world coords)
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    
    # Dictionary to hold results
    wall_footprints = {}
    
    # 2. Loop through each wall
    for wall in walls:
        try:
            # Create the shape
            shape = ifcopenshell.geom.create_shape(settings, wall)
        except Exception as e:
            print(f"Failed to create shape for wall {wall.GlobalId}: {e}")
            continue
        
        # Collect edges as 2D lines
        verts = shape.geometry.verts  # [x0, y0, z0, x1, y1, z1, ...]
        edges = ifcopenshell.util.shape.get_edges(shape.geometry)
        
        lines = []
        for edge in edges:
            coords_2d = []
            for idx in edge:
                x = verts[idx * 3]
                y = verts[idx * 3 + 1]
                coords_2d.append((x, y))
            # Create a LineString for this edge
            lines.append(LineString(coords_2d))
        
        if not lines:
            # No edges for this wall
            continue
        
        # 3. Merge all lines into a single geometry
        multiline = MultiLineString(lines)
        merged = unary_union(multiline)
        
        # 4. Polygonize to form closed polygons
        polygons = list(polygonize(merged))
        if not polygons:
            # Could not form any polygons
            continue
        
        # 5. Pick the largest polygon by area (in case multiple are formed)
        largest_polygon = max(polygons, key=lambda p: p.area)
        
        # 6. Densify the polygon's exterior using substring-based method.
        linestring = largest_polygon.exterior
        densified_points = []
        for i in np.arange(0, linestring.length, sampling_interval):
            segment = substring(linestring, i, i + sampling_interval)
            # Get the boundary (endpoints) of the segment.
            boundary = segment.boundary
            # If the boundary is a Point, add it directly.
            if boundary.geom_type == "Point":
                densified_points.append(boundary)
            # If it's a MultiPoint, add each Point.
            elif boundary.geom_type == "MultiPoint":
                densified_points.extend(list(boundary.geoms))
        
        # Create a MultiPoint from the collected points
        multipoint = MultiPoint(densified_points)
        
        # Convert the MultiPoint to a NumPy array of (x, y) coordinates.
        result = np.array([(pt.x, pt.y) for pt in multipoint.geoms])
        
        # 7. Store result in the dictionary
        wall_footprints[wall.GlobalId] = result
    
    return result


if __name__ == "__main__":
    path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"
    footprints = create_walls_footprints(path, sampling_interval=0.5)
    
    # Plot all wall footprints
    plt.figure(figsize=(15, 10))
    x = footprints[:, 0]
    y = footprints[:, 1]
    plt.scatter(x, y, color="blue")
    plt.title("2D Outlines of All IfcWallStandardCase Elements (Substring Densified as MultiPoint)")
    plt.grid(True)
    plt.legend()
    plt.show()
