from source.transformation_vertical.get_terrain_elevation import get_terrain_elevation, assemble_tiles
from source.transformation_vertical.extract_extents.find_IFC_extent import find_IFC_extent
from source.transformation_vertical.extract_extents.find_CityGML_extent import find_CityGML_extent
from source.transformation_vertical.extract_ifc_centroid import extract_ifc_centroid


def estimate_vertical_offset(citygml_extents, ifc_extents, aided=False, story_mapping={}, terrain_elevation=0.00) -> float:
    # Check min and max z values for CityGML and IFC Stories
    max_z_ifc = max(storey['max_z'] for storey in ifc_extents if storey['max_z'] is not None)
    min_z_ifc = min(storey['min_z'] for storey in ifc_extents if storey['min_z'] is not None)
    max_z_citygml = citygml_extents["max_z"]
    min_z_citygml = citygml_extents["min_z"]

    if aided == False:
        # Check if the vertical extents of the IFC and CityGML files are similar to see if BFBM
        if -2 <= (max_z_ifc - min_z_ifc) - (max_z_citygml - min_z_citygml) <= 2 and aided == False:
            offset = (min_z_citygml + (max_z_citygml - min_z_citygml)/2) - (min_z_ifc + (max_z_ifc - min_z_ifc)/2)
            return round(offset, 2)
        else:
            print("Warning: The vertical extents of the IFC and CityGML files do not match closely enough. Please provide a storey mapping.")
            return None
    else:
        ifc_story = story_mapping["storey_id"]
        ifc_story_min_z = next(item["min_z"] for item in ifc_extents if item["storey_id"] == ifc_story)
        mapped_story_number = story_mapping["storey_number"]
        avg_story_height = (max_z_ifc - min_z_ifc) / len(ifc_extents)
        offset = terrain_elevation - ifc_story_min_z + (mapped_story_number * avg_story_height)
        return round(offset, 2)
    
if __name__ == "__main__":
    ifc_path = "./test_data/ifc/3.002 01-05-0501_EG_transformed_horizontal.ifc"
    ifc_extent = find_IFC_extent(ifc_path=ifc_path)

    citygml_path = "./test_data/citygml/TUM_LoD2_Full_withSurrounds.gml"
    citygml_building_ids = ["DEBY_LOD2_4959457"]
    citygml_extent = find_CityGML_extent(citygml_path=citygml_path, building_ids=citygml_building_ids)

    # Get IFC Centroid
    ifc_centroid = extract_ifc_centroid(ifc_path=ifc_path)

    # Get terrain elevation
    tile_files = [
        "./data/elevation/690_5335.tif",
        "./data/elevation/690_5336.tif",
        "./data/elevation/691_5335.tif",
        "./data/elevation/691_5336.tif",
    ]
    mosaic, transform, meta, nodata = assemble_tiles(tile_files)
    terrain_elevation = get_terrain_elevation(mosaic, transform, ifc_centroid["x"], ifc_centroid["y"], nodata)

    # Define story mapping
    storey_id = "28VlpsEwqevm000040000F"
    storey_number = 0
    story_mapping = {
        "storey_id": storey_id,
        "storey_number": storey_number
    }
    # Estimate vertical offset
    vertical_offset = estimate_vertical_offset(citygml_extents=citygml_extent, ifc_extents=ifc_extent, aided=True, story_mapping=story_mapping, terrain_elevation=terrain_elevation)
    print(f"Estimated vertical offset: {vertical_offset}")