import matplotlib.pyplot as plt
import numpy as np
import ezdxf
from shapely.geometry import Point

def extract_elevation_labels(dxf_path, layer_name) -> np.array:
    """
    Extracts elevation marks and their coordinates as Shapely Points from a DXF file.
    
    Args:
        dxf_path: Path to the DXF file
        layer_name: Name of the layer containing elevation labels
        
    Returns:
        np.array: Array with shape (n, 2) where each row contains [Point, text_label]
    """
    # Open the DXF file and access the modelspace
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Query for all entities in the specified layer
    elevation_entities = msp.query(f'*[layer=="{layer_name}"]')

    elevation_labels = []

    # Iterate through each entity and extract elevation marks
    for entity in elevation_entities:
        if entity.dxftype() == "INSERT":
            x = entity.dxf.insert.x
            y = entity.dxf.insert.y
            
            # Create Shapely Point from coordinates
            point = Point(x, y)
            
            # Try to get text from attached ATTRIB entities
            attribs = list(entity.attribs)
            if attribs:
                text = attribs[0].dxf.text
            else:
                text = ""
                print(f"Warning: INSERT entity at ({x},{y}) has no attributes for text.")
            
            elevation_labels.append([point, text])

    # Convert to numpy array
    return np.array(elevation_labels, dtype=object)

def main():
    dxf_path = "./test_data/dxf/01-05-0501_EG.dxf"
    layer_name = "A_03_HOEHENKOTE"

    elevation_labels = extract_elevation_labels(dxf_path, layer_name)

    # Create a scatter plot for the elevation labels.
    fig, ax = plt.subplots(figsize=(10, 10))
    if elevation_labels.size > 0:
        # Extract x and y coordinates from Shapely Points
        x_coords = [label[0].x for label in elevation_labels]
        y_coords = [label[0].y for label in elevation_labels]
        
        ax.scatter(x_coords, y_coords, c="blue", marker="o", label="Elevation Marks")
        
        # Annotate each point with its text label.
        for point, text in elevation_labels:
            ax.text(point.x, point.y, text, fontsize=10, color="red",
                    verticalalignment="bottom", horizontalalignment="right")
    else:
        ax.text(0.5, 0.5, "No elevation labels found", transform=ax.transAxes, 
                ha="center", va="center")
    
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Elevation Labels from DXF")
    ax.legend()
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    plt.show()

if __name__ == "__main__":
    main()