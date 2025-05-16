# Footprint Creation Imports
from source.transformation_horizontal.create_footprints.create_CityGML_footprint import create_CityGML_footprint
from source.transformation_horizontal.create_footprints.create_IFC_footprint_polygon import create_IFC_footprint_polygon
from source.transformation_horizontal.create_footprints.create_DXF_footprint_polygon import create_DXF_footprint_polygon

# Horizontal Registration Imports
from source.transformation_horizontal.rigid_transformation import Rigid_Transformation
from source.transformation_horizontal.estimate_rigid_transformation import estimate_rigid_transformation, refine_rigid_transformation
from source.transformation_horizontal.detect_features import detect_features, filter_features_by_feature_triangle_area

# Vertical Registration Imports
from source.transformation_vertical.extract_extents.find_CityGML_extent import find_CityGML_extents
from source.transformation_vertical.extract_extents.find_IFC_extent import find_IFC_extents



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

# Transformation of IFC footprint to CityGML footprint and Feature Extraction
print("Transforming IFC Footprint to CityGML Footprint...")
ifc_footprint_transformed = refined_transformation_ifc_to_citygml.transform(ifc_footprint)
ifc_features_transformed = detect_features(footprint=ifc_footprint_transformed, angle_threshold_deg=30)
ifc_features_transformed_filtered = filter_features_by_feature_triangle_area(features=ifc_features_transformed, min_area=15)



## Vertical Registration

# Extract elevation labels from DXF
print("Extracting Elevation Labels from DXF...")
elevation_labels_dxf = extract_elevation_labels(dxf_path=dxf_path, layer_name=dxf_layer)
transformed_elevation_labels_dxf = refined_transformation_dxf_to_ifc.transform_elevation_labels(elevation_labels=elevation_labels_dxf)

# Apply the transformations to the DXF and IFC files
print("Applying Transformations to DXF and IFC files...")
transformed_ifc_file = refined_transformation_ifc_to_citygml.transform_ifc(input_ifc_path=ifc_path, output_ifc_path=".".join(ifc_path.split(".")[:-1]) + "_transformed_horizontal.ifc", z=0) #516.45 for B1. 517.25 for B7
transformed_ifc_file = refined_transformation_ifc_to_citygml.transform_ifc(input_ifc_path=ifc_path, output_ifc_path=".".join(ifc_path.split(".")[:-1]) + "_transformed_vertical.ifc", z=516.53) #516.45 for B1. 517.25 for B7