from source.transformation_vertical.create_sideviews.create_IFC_sideview import create_IFC_sideview
from source.transformation_vertical.create_sideviews.create_CityGML_sideview import create_CityGML_sideview

from source.transformation_horizontal.detect_features import detect_features, filter_features_by_feature_triangle_area
from source.transformation_horizontal.rigid_transformation import Rigid_Transformation
from source.transformation_horizontal.estimate_rigid_transformation import estimate_rigid_transformation, refine_rigid_transformation

import matplotlib.pyplot as plt
from shapely.affinity import translate

citygml_path = "./test_data/citygml/TUM_LoD2_Full_withSurrounds.gml"
ifc_path = "./test_data/ifc/3D_01_05_0501_transformed_horizontal.ifc"

citygml_building_ids = ["DEBY_LOD2_4959457"]
citygml_sideview = create_CityGML_sideview(citygml_path=citygml_path, building_ids=citygml_building_ids)
ifc_sideview = create_IFC_sideview(ifc_path=ifc_path)

# Extract Features using Turning Function for detecting corners in footprints
print("Extracting Features...")
ifc_features = detect_features(footprint=ifc_sideview, angle_threshold_deg=30)
citygml_features = detect_features(footprint=citygml_sideview, angle_threshold_deg=30)
# Filter features by eliminating featrues that span up small triangles with adjacent features to only keep significant features
print("Filtering Features...")
ifc_features_filtered = filter_features_by_feature_triangle_area(features=ifc_features, min_area=15)
citygml_features_filtered = filter_features_by_feature_triangle_area(features=citygml_features, min_area=15)
print(f"Filtered Features: IFC: {len(ifc_features_filtered)}, CityGML: {len(citygml_features_filtered)}")

# --- START: match features by y-offset and pick best inliers ---
tol = 1  # matching tolerance on vertical axis
best_match = None
best_inliers = 0

for ifc_feat in ifc_features_filtered:
    y_ifc = ifc_feat[3]
    for city_feat in citygml_features_filtered:
        y_city = city_feat[3]
        y_offset = y_city - y_ifc
        inliers = 0
        for feat_ifc2 in ifc_features_filtered:
            x2, y2 = feat_ifc2[2], feat_ifc2[3]
            y2_shifted = y2 + y_offset
            # count as inlier if any city feature matches both x and shifted y within tol
            if any(
                abs(city_feat2[2] - x2) <= tol and abs(city_feat2[3] - y2_shifted) <= tol
                for city_feat2 in citygml_features_filtered
            ):
                inliers += 1
        if inliers > best_inliers:
            best_inliers = inliers
            best_match = (ifc_feat, city_feat, y_offset, inliers)

if best_match:
    ifc_feat, city_feat, offset, inliers = best_match
    print(f"Best Y-offset match: offset={offset:.3f}, inliers={inliers}/{len(ifc_features_filtered)}")
    # apply vertical offset to IFC sideview and plot both sideviews
    shifted_ifc = translate(ifc_sideview, xoff=0, yoff=offset)
    plt.figure()
    for poly in citygml_sideview.geoms:
        x, y = poly.exterior.xy
        plt.plot(x, y, color='blue', label='CityGML')
    for poly in shifted_ifc.geoms:
        x, y = poly.exterior.xy
        plt.plot(x, y, color='green', label='IFC (shifted)')
    plt.legend()
    plt.title(f"Sideview Alignment: Y-offset={offset:.3f}, Inliers={inliers}")
    plt.xlabel("Horizontal axis")
    plt.ylabel("Vertical axis")
    plt.show()
else:
    print("No matching Y-offset found.")
# --- END match block ---


