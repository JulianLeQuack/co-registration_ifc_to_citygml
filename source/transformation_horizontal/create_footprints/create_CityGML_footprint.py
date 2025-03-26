import matplotlib.pyplot as plt
import numpy as np

import xml.etree.ElementTree as ET

from shapely.geometry import MultiPoint, Polygon, LineString
from shapely.ops import substring


def create_CityGML_footprint(path_to_CityGML):

    try:
        # Parse the CityGML file
        tree = ET.parse(path_to_CityGML)
        root = tree.getroot()

        # Define namespaces
        ns = {
            'bldg': 'http://www.opengis.net/citygml/building/2.0',
            'gml': 'http://www.opengis.net/gml'
        }

        # Find the first ground surface in the CityGML file
        gs = root.find('.//bldg:GroundSurface', ns)
        if gs is None:
            raise ValueError("No ground surface found in the CityGML file.")

        # Locate the first posList (assuming it's the outer boundary)
        posList = gs.find('.//gml:posList', ns)
        if posList is None:
            raise ValueError("No posList found in the GroundSurface.")

        # Assume coordinates are space separated
        coords = list(map(float, posList.text.split()))

        # Extract only x and y for the 2D polygon
        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 3)]

        # Create a Polygon from the outer boundary points
        polygon = Polygon(points)

        # Densify the Polygon's exterior boundary
        linestring = polygon.exterior
        footprint_densified = MultiPoint()
        for i in np.arange(0, linestring.length, 0.2): #0.2 for dense representation, with 3, the alphashape warning will go away
            s = substring(linestring, i, i + 0.2)
            footprint_densified = footprint_densified.union(s.boundary)

        #Create np array from Multipoint object
        result = np.array([(point.x, point.y) for point in footprint_densified.geoms])

        return result

    except ET.ParseError:
        print(f"Error parsing CityGML file: {path_to_CityGML}")
        return np.array([])
    except ValueError as e:
        print(e)
        return np.array([])
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.array([])


if __name__ == "__main__":
    footprint = create_CityGML_footprint("./test_data/citygml/DEBY_LOD2_4959457.gml")
    plt.figure(figsize=(10,10))
    x = footprint[:, 0]
    y = footprint[:, 1]
    plt.scatter(x, y, color="blue")
    plt.grid(True)
    plt.show()