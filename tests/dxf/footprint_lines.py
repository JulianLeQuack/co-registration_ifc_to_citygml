import sys
import ezdxf
import matplotlib.pyplot as plt

def extract_line_segments(entity):
    """
    Given an entity, extract line segments as tuples of two points (each point as (x, y)).
    If the entity is a LINE, then return its start and end points.
    If it is a POLYLINE or LWPOLYLINE, break it into successive segments.
    """
    segments = []
    dxftype = entity.dxftype()
    if dxftype == 'LINE':
        start = entity.dxf.start
        end = entity.dxf.end
        segments.append(((start[0], start[1]), (end[0], end[1])))
    elif dxftype in ('POLYLINE', 'LWPOLYLINE'):
        try:
            # Try for LWPOLYLINE – returns an iterable of 5-tuples (x, y, start_width, end_width, bulge)
            pts = [pt[:2] for pt in entity]
        except TypeError:
            # Otherwise, use the vertices() method (for 2D POLYLINE)
            pts = [(vertex.dxf.location[0], vertex.dxf.location[1]) for vertex in entity.vertices]
        # Create line segments between successive points
        for i in range(len(pts) - 1):
            segments.append((pts[i], pts[i + 1]))
    return segments

def get_wall_layer_segments(msp, wall_layer_name):
    """
    Find all entities in the modelspace that lie on the given wall layer.
    For each entity that is an ACAD proxy, we try to get its virtual entities
    and then extract line segments from each. For non-proxy LINE/POLYLINE entities,
    we extract segments directly.
    """
    segments = []
    # Query all entities that belong to the specified layer
    wall_entities = msp.query(f'*[layer=="{wall_layer_name}"]')
    for ent in wall_entities:
        if ent.dxftype() == 'ACAD_PROXY_ENTITY':
            # For proxy entities, get the virtual graphic representation:
            try:
                for virtual in ent.virtual_entities():
                    segments.extend(extract_line_segments(virtual))
            except Exception as e:
                print(f"Error processing proxy entity {ent.dxf.handle}: {e}")
        else:
            # For ordinary entities, try to extract segments directly:
            segments.extend(extract_line_segments(ent))
    return segments

def plot_segments(segments, title="Wall Layer Lines"):
    """
    Plot a list of line segments using matplotlib.
    Each segment is a tuple: ((x1,y1), (x2,y2)).
    """
    fig, ax = plt.subplots()
    for (p1, p2) in segments:
        x_values = [p1[0], p2[0]]
        y_values = [p1[1], p2[1]]
        ax.plot(x_values, y_values, 'k-')  # black line
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    plt.show()

def main(filename, wall_layer_name="Walls"):
    try:
        # Read the DXF file
        doc = ezdxf.readfile(filename)
    except IOError:
        print(f"Could not open file: {filename}")
        sys.exit(1)
    except ezdxf.DXFStructureError as e:
        print(f"Invalid or corrupted DXF file: {filename}\nError: {e}")
        sys.exit(2)

    msp = doc.modelspace()
    # Get the line segments for the specified wall layer:
    segments = get_wall_layer_segments(msp, wall_layer_name)
    if not segments:
        print(f"No line segments found on layer '{wall_layer_name}'.")
    else:
        print(f"Found {len(segments)} line segments on layer '{wall_layer_name}'.")
        plot_segments(segments, title=f"DXF Wall Layer: {wall_layer_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_wall_layer.py your_file.dxf [wall_layer_name]")
        sys.exit(1)
    filename = sys.argv[1]
    wall_layer = sys.argv[2] if len(sys.argv) > 2 else "Walls"
    main(filename, wall_layer)
