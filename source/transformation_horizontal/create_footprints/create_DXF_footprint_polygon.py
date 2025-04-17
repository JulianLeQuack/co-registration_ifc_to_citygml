import matplotlib.pyplot as plt
import ezdxf
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import unary_union, polygonize

def to_2d(point):
    """
    Convert a point-like object to a 2D tuple (x, y).
    """
    return (point[0], point[1])

def create_DXF_footprint_polygon(dxf_path, layer_name: str, use_origin_filter: bool = False, origin_threshold: float = 10.0):
    """
    Extracts individual 2D line segments from a DXF file on the specified layer,
    applies a filter to ignore any segment starting or ending at (0,0) that are longer than origin_threshold,
    and then constructs a footprint by performing a unary union followed by polygonization.
    
    All coordinates are converted to 2D (ignoring Z).
    
    Parameters:
        dxf_path (str): Path to the DXF file.
        layer_name (str): Name of the DXF layer to query.
        use_origin_filter (bool): Whether to ignore lines with start or end at (0,0) that exceed the threshold. These lines can appear in DXF files that use blocks.
        origin_threshold (float): Length threshold (in meters) for ignoring lines.
    
    Returns:
        MultiPolygon: A MultiPolygon containing the exterior boundaries of all disjoint footprint areas.
    """
    # Open the DXF file and access the modelspace
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Query for all entities in the specified layer.
    wall_entities = msp.query(f'*[layer=="{layer_name}"]')
    
    line_segments = []  # List to hold individual line segments

    # Iterate through each entity and extract lines
    for entity in wall_entities:
        try:
            for geometry in entity.virtual_entities():
                # Process LINE entities directly
                if geometry.dxftype() == "LINE":
                    start = to_2d(geometry.dxf.start)
                    end = to_2d(geometry.dxf.end)
                    segment = LineString([start, end])
                    line_segments.append(segment)
                
                # Process POLYLINE or LWPOLYLINE by splitting into individual segments
                elif geometry.dxftype() in ["POLYLINE", "LWPOLYLINE"]:
                    try:
                        # Convert each point to 2D by taking only (x, y)
                        points = [to_2d(pt) for pt in geometry.points()]
                    except Exception as e:
                        print(f"Error extracting points from polyline: {e}")
                        continue

                    if len(points) < 2:
                        continue
                    
                    # Create segments from consecutive pairs of 2D points
                    for i in range(len(points) - 1):
                        segment = LineString([points[i], points[i + 1]])
                        line_segments.append(segment)
                
                else:
                    # If additional virtual entities exist, try to process them as well.
                    for sub_geom in geometry.virtual_entities():
                        if sub_geom.dxftype() == "LINE":
                            start = to_2d(sub_geom.dxf.start)
                            end = to_2d(sub_geom.dxf.end)
                            segment = LineString([start, end])
                            line_segments.append(segment)
        except Exception as e:
            print(f"Skipping entity due to error: {e}")
            continue

    # Apply the filter: ignore any segment starting or ending at (0,0) if its length exceeds origin_threshold.
    if use_origin_filter:
        filtered_segments = []
        for seg in line_segments:
            start, end = seg.coords[0], seg.coords[-1]
            if ((start == (0, 0) or end == (0, 0)) and seg.length > origin_threshold):
                continue
            filtered_segments.append(seg)
        line_segments = filtered_segments

    # Merge all line segments and polygonize the network:
    union_lines = unary_union(line_segments)
    poly_list = list(polygonize(union_lines))
    if not poly_list:
        raise ValueError("No polygons formed from the extracted lines.")
    
    # Create a union of the polygons, and extract only the exteriors to form a footprint.
    union_poly = unary_union(poly_list)
    if union_poly.geom_type == "Polygon":
        footprint_mp = MultiPolygon([Polygon(union_poly.exterior)])
    elif union_poly.geom_type == "MultiPolygon":
        footprint_mp = MultiPolygon([Polygon(poly.exterior) for poly in union_poly.geoms])
    else:
        footprint_mp = MultiPolygon([])
    
    return footprint_mp

if __name__ == "__main__":
    # Specify your DXF file path and the target layer name.
    dxf_path = "./test_data/dxf/01-05-0501_EG.dxf"  # Update with your actual file path
    layer_name = "A_09_TRAGDECKE"  # Update if different

    # Extract the footprint as a MultiPolygon
    footprint = create_DXF_footprint_polygon(dxf_path, layer_name, use_origin_filter=False, origin_threshold=10.0)

    # Plot the resulting footprint.
    plt.figure(figsize=(10, 10))
    for poly in footprint.geoms:
        x, y = poly.exterior.xy
        plt.plot(x, y, 'b-', linewidth=1.5)
    
    plt.title("DXF Footprint (Exteriors of Disjoint Polygons)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.show()
