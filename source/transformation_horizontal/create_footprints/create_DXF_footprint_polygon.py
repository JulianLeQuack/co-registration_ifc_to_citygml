import matplotlib.pyplot as plt
import ezdxf
from shapely.geometry import LineString

def to_2d(point):
    """
    Convert a point-like object to a 2D tuple (x, y).
    """
    return (point[0], point[1])

def extract_lines_from_dxf(path_to_dxf, layer_name: str):
    """
    Extracts individual 2D line segments from a DXF file on the specified layer.
    
    It processes:
      - LINE entities (directly extracting start and end points)
      - POLYLINE and LWPOLYLINE entities (splitting them into consecutive segments)
    
    All coordinates are converted to 2D (ignoring Z).
    
    Parameters:
        path_to_dxf (str): Path to the DXF file.
        layer_name (str): Name of the DXF layer to query.
    
    Returns:
        list: A list of Shapely LineString objects representing the 2D line segments.
    """
    # Open the DXF file and access the modelspace
    doc = ezdxf.readfile(path_to_dxf)
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
                    line_segments.append(LineString([start, end]))
                
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
                            line_segments.append(LineString([start, end]))
        except Exception as e:
            print(f"Skipping entity due to error: {e}")
            continue

    return line_segments

if __name__ == "__main__":
    # Specify your DXF file path and the target layer name.
    dxf_path = "./test_data/dxf/01-05-0501_EG.dxf"  # Update with your actual file path
    layer_name = "A_01_TRAGWAND"  # Update with your actual layer name if different

    # Extract the 2D line segments from the DXF file.
    lines = extract_lines_from_dxf(dxf_path, layer_name)

    # Plot the extracted 2D lines using Matplotlib.
    plt.figure(figsize=(10, 10))
    for line in lines:
        x, y = line.xy
        plt.plot(x, y, 'b-', linewidth=1)
    
    plt.title("2D DXF Lines & Polyline Segments")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.show()
