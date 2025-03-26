import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape

import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union, polygonize


def create_slabs_footprints(path_to_ifc, sampling_interval=0.2):
    """
    For each IfcSlab in the IFC file, create a 2D polygon outline and densify it.
    
    Returns:
        A dict {slab_id: numpy_array_of_xy_points, ...}
    """
    ifc_file = ifcopenshell.open(path_to_ifc)
    slabs = ifc_file.by_type("IfcSlab")
    
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    
    slab_footprints = {}
    
    for slab in slabs:
        try:
            shape = ifcopenshell.geom.create_shape(settings, slab)
        except Exception as e:
            print(f"Failed to create shape for slab {slab.GlobalId}: {e}")
            continue
        
        # Collect edges as 2D LineStrings
        verts = shape.geometry.verts  # [x0, y0, z0, x1, y1, z1, ...]
        edges = ifcopenshell.util.shape.get_edges(shape.geometry)
        
        lines = []
        for edge in edges:
            coords_2d = []
            for idx in edge:
                x = verts[idx * 3]
                y = verts[idx * 3 + 1]
                coords_2d.append((x, y))
            # Create a LineString for each edge
            lines.append(LineString(coords_2d))
        
        if not lines:
            continue
        
        # Merge/unify edges into a single geometry
        multiline = MultiLineString(lines)
        merged = unary_union(multiline)
        
        # Polygonize
        polygons = list(polygonize(merged))
        if not polygons:
            continue
        
        # Pick the largest polygon by area (in case there are multiple)
        largest_polygon = max(polygons, key=lambda p: p.area)
        
        # Densify its exterior boundary
        exterior = largest_polygon.exterior
        distances = np.arange(0, exterior.length, sampling_interval)
        densified_coords = [exterior.interpolate(d).coords[0] for d in distances]
        
        # Store in a dictionary keyed by slab's GlobalId
        slab_footprints[slab.GlobalId] = np.array(densified_coords)
    
    return slab_footprints


if __name__ == "__main__":
    path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"
    footprints = create_slabs_footprints(path, sampling_interval=0.5)
    
    # Plot all slab footprints
    plt.figure(figsize=(15, 10))
    
    for slab_id, coords in footprints.items():
        plt.scatter(coords[:, 0], coords[:, 1], s=2, label=slab_id)
    
    plt.title("2D Outlines of All IfcSlab Elements (Densified)")
    plt.grid(True)
    plt.legend()
    plt.show()
