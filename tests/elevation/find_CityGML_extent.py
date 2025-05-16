import xml.etree.ElementTree as ET
from shapely.geometry import Polygon

def find_citygml_z_extent(citygml_path, building_ids=None):
    try:
        tree = ET.parse(citygml_path)
        root = tree.getroot()
        ns_2_0 = {
            'bldg': 'http://www.opengis.net/citygml/building/2.0',
            'gml': 'http://www.opengis.net/gml'
        }
        ns_1_0 = {
            'bldg': 'http://www.opengis.net/citygml/building/1.0',
            'gml': 'http://www.opengis.net/gml'
        }
        min_z, max_z = float("inf"), float("-inf")

        # Try both namespaces for buildings
        def get_buildings(ns):
            return root.findall('.//bldg:Building', ns)

        def get_building_by_id(b_id, ns):
            return root.find(f".//bldg:Building[@gml:id='{b_id}']", ns)

        # Try to detect which namespace is present
        buildings = get_buildings(ns_2_0)
        ns = ns_2_0
        if not buildings:
            buildings = get_buildings(ns_1_0)
            ns = ns_1_0

        if building_ids:
            for b_id in building_ids:
                building = get_building_by_id(b_id, ns)
                if building is None:
                    print(f"Warning: No building found with gml:id '{b_id}'.")
                    continue
                poslists = building.findall('.//gml:posList', ns)
                for poslist in poslists:
                    coords = list(map(float, poslist.text.split()))
                    for i in range(2, len(coords), 3):  # Extract Z-coordinates
                        z = coords[i]
                        min_z = min(min_z, z)
                        max_z = max(max_z, z)
        else:
            for building in buildings:
                poslists = building.findall('.//gml:posList', ns)
                for poslist in poslists:
                    coords = list(map(float, poslist.text.split()))
                    for i in range(2, len(coords), 3):  # Extract Z-coordinates
                        z = coords[i]
                        min_z = min(min_z, z)
                        max_z = max(max_z, z)

        if min_z == float("inf") or max_z == float("-inf"):
            print("No valid Z-coordinates found in the CityGML file.")
            return None, None

        return round(min_z, 2), round(max_z, 2)

    except ET.ParseError as e:
        print(f"Error parsing CityGML file: {citygml_path}, Error: {e}")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

if __name__ == "__main__":
    # citygml_path = "./test_data/citygml/TUM_LoD2_Full_withSurrounds.gml"
    # building_ids = ["DEBY_LOD2_4959457"]  # Optional: Specify building IDs or set to None
    citygml_path = "./test_data/citygml/690_5336.gml"
    building_ids = ["DEBY_LOD2_108580336"]  # Optional: Specify building IDs or set to None
    min_z, max_z = find_citygml_z_extent(citygml_path, building_ids)
    if min_z is not None and max_z is not None:
        print(f"Z-Extent of the CityGML model: Min Z = {min_z}, Max Z = {max_z}")
    else:
        print("Failed to calculate Z-extent.")