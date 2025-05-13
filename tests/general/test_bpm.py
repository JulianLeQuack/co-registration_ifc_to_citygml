from source.transformation_horizontal.create_footprints.create_IFC_footprint_polygon import create_IFC_footprint_polygon, extract_classes, extract_building_storeys
from source.transformation_horizontal.create_footprints.create_CityGML_footprint import create_CityGML_footprint, extract_building_ids
from source.transformation_horizontal.create_footprints.create_DXF_footprint_polygon import create_DXF_footprint_polygon, extract_layers

from source.transformation_horizontal.detect_features import detect_features, filter_features_by_feature_triangle_area
from source.transformation_horizontal.rigid_transformation import Rigid_Transformation
from source.transformation_horizontal.estimate_rigid_transformation import estimate_rigid_transformation, refine_rigid_transformation
import matplotlib.pyplot as plt
import numpy as np

# input files
citygml_path = "./test_data/citygml/TUM_LoD2_Full_withSurrounds.gml"
citygml_building_ids = ["DEBY_LOD2_4959457"]

ifc_path = "./test_data/ifc/bpm/bpm_b1_entrance_hall.ifc"
output_ifc_path = "./test_data/ifc/bpm/bpm_b1_entrance_hall_transformed.ifc"
ifc_building_storeys = extract_building_storeys(ifc_path)[0] # bpm only has one storey
ifc_class = "IfcWall" # slabs not included in model
b1_reference_elevation = 516.45

dxf_path = "./test_data/dxf/01-05-0501_EG.dxf"
dxf_layer_walls = "A_01_TRAGWAND"
dxf_layer_slabs = "A_09_TRAGDECKE"



# create footprints
print("Creating Footprints...")
citygml_footprint = create_CityGML_footprint(citygml_path=citygml_path, building_ids=citygml_building_ids)
ifc_footprint = create_IFC_footprint_polygon(ifc_path=ifc_path, ifc_type=ifc_class, building_storeys=[ifc_building_storeys])
dxf_footprint_walls = create_DXF_footprint_polygon(dxf_path=dxf_path, layer_name=dxf_layer_walls, use_origin_filter=True, origin_threshold=10.0)
dxf_footprint_slabs = create_DXF_footprint_polygon(dxf_path=dxf_path, layer_name=dxf_layer_slabs)


# plot the footprints in subplots
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
for poly in citygml_footprint.geoms:
    x, y = poly.exterior.xy
    plt.plot(x, y, color='blue', label='CityGML')
plt.title("CityGML Footprint")
plt.subplot(2, 2, 2)
for poly in ifc_footprint.geoms:
    x, y = poly.exterior.xy
    plt.plot(x, y, color='red', label='IFC')
plt.title("IFC Footprint")
plt.subplot(2, 2, 3)
for poly in dxf_footprint_walls.geoms:
    x, y = poly.exterior.xy
    plt.plot(x, y, color='green', label='DXF Walls')
plt.title("DXF Walls Footprint")
plt.subplot(2, 2, 4)
for poly in dxf_footprint_slabs.geoms:
    x, y = poly.exterior.xy
    plt.plot(x, y, color='orange', label='DXF Slabs')
plt.title("DXF Slabs Footprint")
plt.tight_layout()
plt.show()


# extract features
print("Extracting Features...")
ifc_features = detect_features(footprint=ifc_footprint, angle_threshold_deg=30)
citygml_features = detect_features(footprint=citygml_footprint, angle_threshold_deg=30)
dxf_features_slabs = detect_features(footprint=dxf_footprint_slabs, angle_threshold_deg=30)
dxf_features_walls = detect_features(footprint=dxf_footprint_walls, angle_threshold_deg=30)
# Filter features by eliminating featrues that span up small triangles with adjacent features to only keep significant features
print("Filtering Features...")
ifc_features_filtered = filter_features_by_feature_triangle_area(features=ifc_features, min_area=5)
citygml_features_filtered = filter_features_by_feature_triangle_area(features=citygml_features, min_area=15)
dxf_features_filtered_slabs = filter_features_by_feature_triangle_area(features=dxf_features_slabs, min_area=15)
dxf_features_filtered_walls = filter_features_by_feature_triangle_area(features=dxf_features_walls, min_area=5)
print(f"Filtered Features: IFC: {len(ifc_features_filtered)}, CityGML: {len(citygml_features_filtered)}, DXF Walls: {len(dxf_features_filtered_walls)}, DXF Slabs: {len(dxf_features_filtered_slabs)}")



# estimate horizontal transformation
print("Estimating Rigid Transformation of Floor Plan Slabs to CityGML...")
# estimate rigid transformation
rough_transformation_dxf_to_citygml, inlier_pairs_dxf_to_citygml = estimate_rigid_transformation(
    source_features=dxf_features_filtered_slabs,
    target_features=citygml_features_filtered,
    distance_tol=1,
    angle_tol_deg=45,
)

refined_transformation_dxf_to_citygml = refine_rigid_transformation(inlier_pairs=inlier_pairs_dxf_to_citygml)
print(f"Transformation for DXF to CityGML: {refined_transformation_dxf_to_citygml}")


dxf_footprint_walls_transformed = refined_transformation_dxf_to_citygml.transform(dxf_footprint_walls)

# plot the registered features
plt.figure(figsize=(10, 8))
# Plot CityGML footprint
for poly in citygml_footprint.geoms:
    x, y = poly.exterior.xy
    plt.plot(x, y, color='blue', label='CityGML')
# Plot CityGML features
plt.scatter(citygml_features_filtered[:, 2], citygml_features_filtered[:, 3], 
            color='blue', s=50, marker='o', label='CityGML Features')

# Plot transformed DXF walls
for poly in dxf_footprint_walls_transformed.geoms:
    x, y = poly.exterior.xy
    plt.plot(x, y, color='red', label='DXF Walls')
# Plot transformed DXF features
dxf_features_filtered_walls_transformed = refined_transformation_dxf_to_citygml.transform_features(dxf_features_filtered_walls)
plt.scatter(dxf_features_filtered_walls_transformed[:, 2], dxf_features_filtered_walls_transformed[:, 3], 
            color='red', s=50, marker='x', label='DXF Features')

# Fix legend to show each item only once
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title("Registered Features (DXF to CityGML)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axis("equal")
plt.tight_layout()
plt.show()


dxf_features_filtered_walls_transformed = refined_transformation_dxf_to_citygml.transform_features(dxf_features_filtered_walls)


# estimate ifc to dxf transformation
print("Estimating Rigid Transformation of IFC to DXF...")
# estimate rigid transformation
rough_transformation_ifc_to_dxf, inlier_pairs_ifc_to_dxf = estimate_rigid_transformation(
    source_features=ifc_features_filtered,
    target_features=dxf_features_filtered_walls_transformed,
    distance_tol=1,
    angle_tol_deg=45,
)

refined_transformation_ifc_to_dxf = refine_rigid_transformation(inlier_pairs=inlier_pairs_ifc_to_dxf)
print(f"Transformation for IFC to DXF: {refined_transformation_ifc_to_dxf}")

# apply the transformation to the IFC footprint
ifc_footprint_transformed = refined_transformation_ifc_to_dxf.transform(ifc_footprint)


# plot the registered features
plt.figure(figsize=(10, 8))
# Plot DXF walls (transformed to CityGML space)
for poly in dxf_footprint_walls_transformed.geoms:
    x, y = poly.exterior.xy
    plt.plot(x, y, color='blue', label='DXF Walls')
# Plot DXF features
plt.scatter(dxf_features_filtered_walls_transformed[:, 2], dxf_features_filtered_walls_transformed[:, 3], 
            color='blue', s=50, marker='x', label='DXF Features')

# Plot transformed IFC
for poly in ifc_footprint_transformed.geoms:
    x, y = poly.exterior.xy
    plt.plot(x, y, color='red', label='IFC')
# Transform and plot IFC features
ifc_features_transformed = refined_transformation_ifc_to_dxf.transform_features(ifc_features_filtered)
plt.scatter(ifc_features_transformed[:, 2], ifc_features_transformed[:, 3], 
            color='red', s=50, marker='o', label='IFC Features')

# Plot CityGML
for poly in citygml_footprint.geoms:
    x, y = poly.exterior.xy
    plt.plot(x, y, color='green', label='CityGML')
# Plot CityGML features
plt.scatter(citygml_features_filtered[:, 2], citygml_features_filtered[:, 3], 
            color='green', s=50, marker='^', label='CityGML Features')

# Fix legend to show each item only once
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title("All Registered Features")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axis("equal")
plt.tight_layout()
plt.show()


# apply the transformation to the IFC footprint
ifc_footprint_transformed = refined_transformation_ifc_to_dxf.transform_ifc(input_ifc_path=ifc_path, output_ifc_path=output_ifc_path, z=b1_reference_elevation)