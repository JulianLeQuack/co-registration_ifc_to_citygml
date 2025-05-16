import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET

from shapely.geometry import Polygon, MultiPolygon

def create_CityGML_footprint(citygml_path, building_ids: list) -> MultiPolygon:
    """
    Parses a CityGML file and returns a MultiPolygon.
    If building_ids (a list of strings) is provided, only footprints for those buildings
    (matched via the 'gml:id' attribute of bldg:Building elements) are returned.
    Otherwise, footprints from all ground surfaces in the file are processed.
    :param citygml_path: Path to the CityGML file
    :param building_ids: List of building IDs to process
    :return: MultiPolygon of the building footprints
    """
    try:
        # Parse the CityGML file
        tree = ET.parse(citygml_path)
        root = tree.getroot()

        # Define namespaces for 2.0 and 1.0
        ns_2_0 = {
            'bldg': 'http://www.opengis.net/citygml/building/2.0',
            'gml': 'http://www.opengis.net/gml'
        }
        ns_1_0 = {
            'bldg': 'http://www.opengis.net/citygml/building/1.0',
            'gml': 'http://www.opengis.net/gml'
        }

        # Try to detect which namespace is present
        buildings = root.findall('.//bldg:Building', ns_2_0)
        ns = ns_2_0
        if not buildings:
            buildings = root.findall('.//bldg:Building', ns_1_0)
            ns = ns_1_0

        polygons = []
        if building_ids:
            # Process ground surfaces only from specified buildings
            for b_id in building_ids:
                building = root.find(f".//bldg:Building[@gml:id='{b_id}']", ns)
                if building is None:
                    print(f"Warning: No building found with gml:id '{b_id}'.")
                    continue
                # Get GroundSurface elements within this building
                ground_surfaces = building.findall(".//bldg:GroundSurface", ns)
                if not ground_surfaces:
                    print(f"Warning: No GroundSurface found in building '{b_id}'.")
                for gs in ground_surfaces:
                    posList = gs.find('.//gml:posList', ns)
                    if posList is None:
                        print("Warning: A GroundSurface element without a posList was found; skipping it.")
                        continue

                    # Coordinates should be space separated and form triplets (x y z)
                    coords = list(map(float, posList.text.split()))
                    # Extract x and y for the 2D polygon
                    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 3)]
                    if len(points) >= 3:
                        poly = Polygon(points)
                        if poly.is_valid:
                            polygons.append(poly)
                        else:
                            print("Warning: An invalid polygon was created; skipping it.")
                    else:
                        print("Warning: Not enough points to form a polygon; skipping.")
        else:
            # Fall back to processing all ground surfaces in the file
            ground_surfaces = root.findall('.//bldg:GroundSurface', ns)
            if not ground_surfaces:
                raise ValueError("No ground surface found in the CityGML file.")
            for gs in ground_surfaces:
                posList = gs.find('.//gml:posList', ns)
                if posList is None:
                    print("Warning: A GroundSurface element without a posList was found; skipping it.")
                    continue

                coords = list(map(float, posList.text.split()))
                points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 3)]
                if len(points) >= 3:
                    poly = Polygon(points)
                    if poly.is_valid:
                        polygons.append(poly)
                    else:
                        print("Warning: An invalid polygon was created; skipping it.")
                else:
                    print("Warning: Not enough points to form a polygon; skipping.")

        return MultiPolygon(polygons) if polygons else MultiPolygon([])

    except ET.ParseError:
        print(f"Error parsing CityGML file: {citygml_path}")
        return MultiPolygon([])
    except ValueError as e:
        print(e)
        return MultiPolygon([])
    except Exception as e:
        print(f"An error occurred: {e}")
        return MultiPolygon([])

def extract_building_ids(citygml_path: str) -> list:
    """_
    Extracts building IDs from a CityGML file.
    :param citygml_path: Path to the CityGML file
    :return: List of building IDs (no None entries)
    """
    try:
        tree = ET.parse(citygml_path)
        root = tree.getroot()
        ns = {
            'bldg': 'http://www.opengis.net/citygml/building/2.0',
            'gml': 'http://www.opengis.net/gml'
        }

        ids = []
        # find all <bldg:Building> elements
        for b in root.findall('.//bldg:Building', ns):
            # attribute key is the full URI wrapped in {}
            key = f"{{{ns['gml']}}}id"
            bid = b.attrib.get(key)
            if bid:
                ids.append(bid)
        return ids

    except ET.ParseError:
        print(f"Error parsing CityGML file: {citygml_path}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


if __name__ == "__main__":
    # Use a test file and optionally a list of building IDs 
    # For example, supply a list of building IDs: ['B1', 'B2']
    citygml_path = "./test_data/citygml/690_5336.gml"
    # building_ids = ["DEBY_LOD2_108580336"]
    building_ids = []

    print("Extracting building IDs from CityGML...")
    # building_ids = extract_building_ids("./test_data/citygml/TUM_LoD2_Full_withSurrounds.gml")
    print(f"Building IDs: {building_ids}")
    # footprint = create_CityGML_footprint("./test_data/citygml/TUM_LoD2_Full_withSurrounds.gml", ['DEBY_LOD2_4959793', 'DEBY_LOD2_4959323', 'DEBY_LOD2_4959321', 'DEBY_LOD2_4959324', 'DEBY_LOD2_4959459', 'DEBY_LOD2_4959322', 'DEBY_LOD2_4959458'])
    footprint = create_CityGML_footprint(citygml_path=citygml_path, building_ids=building_ids)
    print(f"Output Type: {type(footprint)}")
    plt.figure(figsize=(10,10))
    if not footprint.is_empty:
        for poly in footprint.geoms:
            x, y = poly.exterior.xy
            plt.plot(x, y, color="blue")
    plt.grid(True)
    plt.show()