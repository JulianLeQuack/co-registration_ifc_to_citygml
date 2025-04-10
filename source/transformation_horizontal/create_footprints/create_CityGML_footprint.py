import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET

from shapely.geometry import Polygon, MultiPolygon

def create_CityGML_footprint(path_to_CityGML):
    """
    Parses a CityGML file and returns a MultiPolygon. It searches for all ground surface elements,
    extracts the coordinates from their posList, and forms a Polygon from each.
    If only one polygon is found, it is wrapped in a MultiPolygon.
    """
    try:
        # Parse the CityGML file
        tree = ET.parse(path_to_CityGML)
        root = tree.getroot()

        # Define namespaces
        ns = {
            'bldg': 'http://www.opengis.net/citygml/building/2.0',
            'gml': 'http://www.opengis.net/gml'
        }

        # Find all ground surfaces in the CityGML file
        ground_surfaces = root.findall('.//bldg:GroundSurface', ns)
        if not ground_surfaces:
            raise ValueError("No ground surface found in the CityGML file.")

        polygons = []
        for gs in ground_surfaces:
            # Locate the first posList (assuming it's the outer boundary)
            posList = gs.find('.//gml:posList', ns)
            if posList is None:
                print("Warning: A GroundSurface element without a posList was found; skipping it.")
                continue

            # Assume coordinates are space separated and form triplets (x y z)
            coords = list(map(float, posList.text.split()))
            # Extract only x and y for the 2D polygon
            points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 3)]

            # Create polygon if points are sufficient
            if len(points) >= 3:
                poly = Polygon(points)
                if poly.is_valid:
                    polygons.append(poly)
                else:
                    print("Warning: An invalid polygon was created; skipping it.")
            else:
                print("Warning: Not enough points to form a polygon; skipping.")

        # Wrap the polygons into a MultiPolygon
        if polygons:
            return MultiPolygon(polygons)
        else:
            return MultiPolygon([])

    except ET.ParseError:
        print(f"Error parsing CityGML file: {path_to_CityGML}")
        return MultiPolygon([])
    except ValueError as e:
        print(e)
        return MultiPolygon([])
    except Exception as e:
        print(f"An error occurred: {e}")
        return MultiPolygon([])


if __name__ == "__main__":
    # Use a test file (change path as needed)
    #footprint = create_CityGML_footprint("./test_data/citygml/DEBY_LOD2_4959457.gml")
    footprint = create_CityGML_footprint("./test_data/citygml/TUM_LoD2_Full_withSurrounds.gml")
    print(type(footprint))
    plt.figure(figsize=(10,10))
    # If you would like to see each polygon, iterate through them:
    if not footprint.is_empty:
        for poly in footprint.geoms:
            x, y = poly.exterior.xy
            plt.plot(x, y, color="blue")
    plt.grid(True)
    plt.show()