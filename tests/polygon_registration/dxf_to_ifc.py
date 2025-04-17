import matplotlib.pyplot as plt

from source.transformation_horizontal.create_footprints.create_IFC_footprint_polygon import create_IFC_footprint_polygon
from source.transformation_horizontal.create_footprints.create_DXF_footprint_polygon import create_DXF_footprint_polygon

from source.transformation_horizontal.estimate_rigid_transformation import estimate_rigid_transformation, refine_rigid_transformation
from source.transformation_horizontal.detect_features import detect_features, filter_features_by_feature_triangle_area
from source.transformation_horizontal.rigid_transformation import Rigid_Transformation

ifc_path = "./test_data/ifc/3.003 01-05-0507_EG.ifc"
dxf_path = "./test_data/dxf/01-05-0507_EG.1.dxf"

ifc_footprint = create_IFC_footprint_polygon(ifc_path=ifc_path, ifc_type="IfcSlab")
dxf_footprint = create_DXF_footprint_polygon(dxf_path=dxf_path, layer_name="A_09_TRAGDECKE")
ifc_features = detect_features(footprint=ifc_footprint, angle_threshold_deg=30)
dxf_features = detect_features(footprint=dxf_footprint, angle_threshold_deg=30)
dxf_features_filtered = filter_features_by_feature_triangle_area(features=dxf_features, min_area=10)
ifc_features_filtered = filter_features_by_feature_triangle_area(features=ifc_features, min_area=10)

rough_trans, inliers = estimate_rigid_transformation(source_features=dxf_features_filtered, target_features=ifc_features_filtered, distance_tol=1, angle_tol_deg=45)
refined_trans = refine_rigid_transformation(inlier_pairs=inliers)

print(f"Transformation: {refined_trans}\nNumber of features (dxf/ifc): {len(dxf_features_filtered)}/{len(ifc_features_filtered)}\nInliers: {len(inliers)}")

refined_trans.transform_shapely_polygon(dxf_footprint)

# Plot to verify alignment
fig, ax = plt.subplots()

def plot_geom(geom, **kwargs):
    geoms = geom.geoms if hasattr(geom, 'geoms') else [geom]
    for g in geoms:
        x, y = g.exterior.xy
        ax.plot(x, y, **kwargs)

plot_geom(ifc_footprint, color='blue', linewidth=2, label='IFC footprint')
plot_geom(dxf_footprint, color='red', linestyle='--', linewidth=2,
          label='DXF footprint (transformed)')

ax.set_aspect('equal')
ax.legend()
ax.set_title('IFC vs Transformed DXF Footprints')
plt.show()

