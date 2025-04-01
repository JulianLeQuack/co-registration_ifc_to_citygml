import matplotlib.pyplot as plt
import numpy as np

import ezdxf


def create_DXF_footprint(path_to_DXF, wall_layer_name: str):

    points = []

    # Parse DXF file and create Modelspace
    file = ezdxf.readfile(path_to_DXF)
    msp = file.modelspace()

    # Query CAD Objects in layer "A_01_TRAGWAND"
    wall_entities = msp.query(f'*[layer=="{wall_layer_name}"]')

    # Iterate over all entites in layer
    for entity in wall_entities:
        # Extract virtual entites from ACAD_PROXY_ENTITy
        object = entity.virtual_entities()
        # Extract entities from virtual entities
        for geometry in object:
            # Extract POLYLINE entities
            if geometry.dxftype() == "POLYLINE":
                # Ectract Vertices list from Polylines
                vertices = geometry.points()
                # Extract Vertices from Vertices List
                for vertex in vertices:
                    points.append(vertex)

    # Check if points were extracted
    if not points:
        raise ValueError("No ground surface points found in the DXF file.")

    return np.array(points)


if __name__ == "__main__":
    footprint = create_DXF_footprint("./test_data/dxf/01-05-0501_EG.dxf", "A_01_TRAGWAND")
    plt.figure(figsize=(15,10))
    x = footprint[:, 0]
    y = footprint[:, 1]
    plt.scatter(x, y)
    plt.grid(True)
    plt.show()