import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

def create_CityGML_sideview(citygml_path, building_ids: list) -> MultiPolygon:
    try:
        tree = ET.parse(citygml_path)
        root = tree.getroot()
        ns = {
            'bldg': 'http://www.opengis.net/citygml/building/2.0',
            'gml': 'http://www.opengis.net/gml'
        }
        polygons = []
        if building_ids:
            for b_id in building_ids:
                building = root.find(f".//bldg:Building[@gml:id='{b_id}']", ns)
                if building is None:
                    print(f"Warning: No building found with gml:id '{b_id}'.")
                    continue
                elements = building.findall(".//bldg:WallSurface", ns)
                if not elements:
                    print(f"Warning: No BuildingPart found in building '{b_id}'.")
                for element in elements:
                    poslist = element.find('.//gml:posList', ns)
                    if poslist is None:
                        print("Warning: A BuildingPart element without a posList was found; skipping it.")
                        continue
                    coords = list(map(float, poslist.text.split()))
                    # side‐view: take Y,Z
                    points = [(coords[i + 1], coords[i + 2]) for i in range(0, len(coords), 3)]
                    if len(points) >= 3:
                        poly = Polygon(points)
                        if poly.is_valid:
                            polygons.append(poly)
                        else:
                            print("Warning: An invalid polygon was created; skipping it.")
                    else:
                        print("Warning: Not enough points to form a polygon; skipping.")
        else:
            elements = root.findall('.//bldg:BuildingPart', ns)
            for element in elements:
                poslist = element.find('.//gml:posList', ns)
                if poslist is None:
                    print("Warning: A BuildingPart element without a posList was found; skipping it.")
                    continue
                coords = list(map(float, poslist.text.split()))
                # side‐view: take Y,Z
                points = [(coords[i + 1], coords[i + 2]) for i in range(0, len(coords), 3)]
                if len(points) >= 3:
                    poly = Polygon(points)
                    if poly.is_valid:
                        polygons.append(poly)
                    else:
                        print("Warning: An invalid polygon was created; skipping it.")
                else:
                    print("Warning: Not enough points to form a polygon; skipping.")
        
        # union all wall polygons into disjoint MultiPolygon
        if not polygons:
            return MultiPolygon([])
        unioned = unary_union(polygons)
        if unioned.geom_type == "Polygon":
            return MultiPolygon([unioned])
        elif unioned.geom_type == "MultiPolygon":
            return unioned
        else:
            print(f"Unexpected geometry type: {unioned.geom_type}")
            return MultiPolygon([])

    except ET.ParseError as e:
        print(f"Error parsing CityGML file: {citygml_path}, Error: {e}")
        return MultiPolygon([])
    except ValueError as e:
        print(e)
        return MultiPolygon([])
    except Exception as e:
        print(f"An error occurred: {e}")
        return MultiPolygon([])
    
if __name__ == "__main__":
    # Example usage
    citygml_path = "./test_data/citygml/TUM_LoD2_Full_withSurrounds.gml"
    building_ids = ["DEBY_LOD2_4959457", "DEBY_LOD2_4959323"]
    sideview = create_CityGML_sideview(citygml_path, building_ids)
    
    # Plotting the sideview
    if not sideview.is_empty:
        for poly in sideview.geoms:
            y, z = poly.exterior.xy
            plt.plot(y, z, color='blue')
        plt.title("CityGML Sideview (Y–Z)")
        plt.xlabel("Y Coordinate")
        plt.ylabel("Z Coordinate")
        plt.show()
    else:
        print("No valid polygons to display.")