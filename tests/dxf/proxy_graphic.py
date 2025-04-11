import matplotlib.pyplot as plt
import ezdxf
from shapely.geometry import LineString

def to_2d(point):
    """Convert a point-like object to a 2D tuple (x, y)."""
    return (point[0], point[1])

def extract_lines_from_dxf_using_proxy(path_to_dxf, layer_name: str):
    """
    Extracts individual 2D line segments using proxy graphics.
    
    This function attempts to load the proxy graphic from each ACADProxyEntity and,
    if available, iterates over its sub-entities.
    
    Parameters:
        path_to_dxf (str): Path to the DXF file.
        layer_name (str): Name of the DXF layer to query.
    
    Returns:
        list: A list of Shapely LineString objects representing the 2D segments.
    """
    doc = ezdxf.readfile(path_to_dxf)
    msp = doc.modelspace()

    wall_entities = msp.query(f'*[layer=="{layer_name}"]')
    line_segments = []
    
    for entity in wall_entities:
        # Check if the entity is a proxy entity.
        if entity.dxftype() == "ACAD_PROXY_ENTITY":
            try:
                proxy_graphics = entity.load_proxy_graphic()
            except Exception as e:
                print(f"Error loading proxy graphic from {entity.dxftype()}: {e}")
                continue

            # If no proxy graphic data is available, skip this entity:
            if proxy_graphics is None:
                print("No proxy graphic found for this entity.")
                continue

            # If proxy_graphics is not an iterable (but a single entity), wrap it in a list.
            if not hasattr(proxy_graphics, '__iter__'):
                proxy_graphics = [proxy_graphics]

            for pg in proxy_graphics:
                if pg.dxftype() == "LINE":
                    start = to_2d(pg.dxf.start)
                    end = to_2d(pg.dxf.end)
                    line_segments.append(LineString([start, end]))
                elif pg.dxftype() in ["POLYLINE", "LWPOLYLINE"]:
                    try:
                        points = [to_2d(pt) for pt in pg.points()]
                    except Exception as e:
                        print(f"Error extracting points from polyline: {e}")
                        continue

                    if len(points) < 2:
                        continue

                    for i in range(len(points) - 1):
                        line_segments.append(LineString([points[i], points[i + 1]]))
                    if getattr(pg, 'closed', False):
                        line_segments.append(LineString([points[-1], points[0]]))
                # Additional entity types can be processed here as needed.
        else:
            # Optionally, handle non-proxy entities if desired.
            continue

    return line_segments

if __name__ == "__main__":
    dxf_path = "./test_data/dxf/01-05-0501_EG.dxf"  # Update with your actual file path
    layer_name = "A_01_TRAGWAND"  # Update with your actual layer name if needed

    # Extract the 2D line segments using the proxy graphic directly.
    lines = extract_lines_from_dxf_using_proxy(dxf_path, layer_name)

    # Plot the extracted 2D lines using Matplotlib.
    plt.figure(figsize=(10, 10))
    for line in lines:
        x, y = line.xy
        plt.plot(x, y, 'b-', linewidth=0.1)

    plt.title("2D DXF Proxy Graphics (Lines & Polyline Segments)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.show()
