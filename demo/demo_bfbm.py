# Footprint Creation Imports
from source.transformation_horizontal.create_footprints.create_CityGML_footprint import create_CityGML_footprint
from source.transformation_horizontal.create_footprints.create_IFC_footprint_polygon import create_IFC_footprint_polygon

# Horizontal Registration Imports
from source.transformation_horizontal.rigid_transformation import Rigid_Transformation
from source.transformation_horizontal.estimate_rigid_transformation import estimate_rigid_transformation, refine_rigid_transformation
from source.transformation_horizontal.detect_features import detect_features, filter_features_by_feature_triangle_area

# Vertical Registration Imports
from source.transformation_vertical.extract_extents.find_CityGML_extent import find_CityGML_extent
from source.transformation_vertical.extract_extents.find_IFC_extent import find_IFC_extent
from source.transformation_vertical.estimate_vertical_offset import estimate_vertical_offset



# Input Data Paths


citygml_path = "./test_data/citygml/690_5336.gml"

ifc_path = "./test_data/ifc/3D_01_05_0501.ifc"

# Model footprint settings

citygml_building_ids = ["DEBY_LOD2_108580336"]

building_storeys = ["100"]  # List of building storeys to be used for the IFC footprint
ifc_type = "IfcSlab"

# Create Footprints
print("Creating Footprints...")
ifc_footprint = create_IFC_footprint_polygon(ifc_path=ifc_path, ifc_type=ifc_type, building_storeys=building_storeys)
citygml_footprint = create_CityGML_footprint(citygml_path=citygml_path, building_ids=citygml_building_ids)

# Extract Features using Turning Function for detecting corners in footprints
print("Extracting Features...")
ifc_features = detect_features(footprint=ifc_footprint, angle_threshold_deg=30)
citygml_features = detect_features(footprint=citygml_footprint, angle_threshold_deg=30)

# Filter features by eliminating featrues that span up small triangles with adjacent features to only keep significant features
print("Filtering Features...")
ifc_features_filtered = filter_features_by_feature_triangle_area(features=ifc_features, min_area=5)
citygml_features_filtered = filter_features_by_feature_triangle_area(features=citygml_features, min_area=5)
print(f"Filtered Features: IFC: {len(ifc_features_filtered)}, CityGML: {len(citygml_features_filtered)}")

# Estimate Rigid Transformation for IFC to CityGML
print("Estimating Rigid Transformation for IFC Footprint...")
rough_transformation_ifc_to_citygml, inlier_pairs_ifc_to_citygml = estimate_rigid_transformation(source_features=ifc_features_filtered, target_features=citygml_features_filtered, distance_tol=1, angle_tol_deg=45)
refined_transformation_ifc_to_citygml = refine_rigid_transformation(inlier_pairs=inlier_pairs_ifc_to_citygml)
print(f"Transformation for IFC to CityGML: {refined_transformation_ifc_to_citygml}")

# Export the IFC file after horizontal transformation
transformed_ifc_path = ".".join(ifc_path.split(".")[:-1]) + "_transformed.ifc"
print(f"Exporting transformed IFC file to: {transformed_ifc_path}")
transformed_ifc_file = refined_transformation_ifc_to_citygml.transform_ifc(input_ifc_path=ifc_path, output_ifc_path=transformed_ifc_path) #516.45 for B1. 517.25 for B7

# Extract the extents of the CityGML and IFC files
citygml_extent = find_CityGML_extent(citygml_path=citygml_path, building_ids=citygml_building_ids)
ifc_extent = find_IFC_extent(ifc_path=transformed_ifc_path)

vertical_offset = estimate_vertical_offset(citygml_extents=citygml_extent, ifc_extents=ifc_extent)


# Export IFC file after vertical transformation
refined_transformation_ifc_to_citygml.offset_ifc(input_ifc_path=transformed_ifc_path, output_ifc_path=transformed_ifc_path, z=vertical_offset)
print(f"Vertical Offset: {vertical_offset}")
print("Transformation completed successfully.")