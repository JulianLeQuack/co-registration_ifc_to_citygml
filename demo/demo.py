# Footprint Creation Imports
from source.transformation_horizontal.create_footprints.create_CityGML_footprint import create_CityGML_footprint
from source.transformation_horizontal.create_footprints.create_IFC_footprint_polygon import create_IFC_footprint_polygon
from source.transformation_horizontal.create_footprints.create_DXF_footprint_polygon import create_DXF_footprint_polygon

# Horizontal Registration Imports
from source.transformation_horizontal.rigid_transformation import Rigid_Transformation
from source.transformation_horizontal.estimate_rigid_transformation import estimate_rigid_transformation, refine_rigid_transformation
from source.transformation_horizontal.detect_features import detect_features, filter_features_by_feature_triangle_area

# Vertical Registration Imports
from source.transformation_vertical.extract_elevation_labels import extract_elevation_labels


# Input Data Paths
ifc_path = "./demo/data/3.002 01-05-0501_EG.ifc"
dxf_path = "./demo/data/3.002 01-05-0501_EG.dxf"
citygml_path = "./demo/data/3.002 01-05-0501_EG.gml"
output_path = "./demo/data/"

# Create Footprints
create_IFC_footprint_polygon(ifc_path, output_path)
create_DXF_footprint_polygon(dxf_path, output_path)
create_CityGML_footprint(citygml_path, output_path)

# Extract Features
ifc_features = detect_features(ifc_path, )