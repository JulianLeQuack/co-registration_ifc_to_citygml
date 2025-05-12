import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from source.transformation_horizontal.create_footprints.create_IFC_footprint_polygon import (
    create_IFC_footprint_polygon,
)
from source.transformation_horizontal.detect_features import (
    detect_features,
    filter_features_by_feature_edge_length,
    filter_features_by_feature_triangle_area,
)

# 1. Load the IFC footprint
ifc_path = "./test_data/ifc/3D_01_05_0507.ifc"
ifc_building_storeys = ["100"]
footprint_ifc = create_IFC_footprint_polygon(
    ifc_path=ifc_path, ifc_type="IfcSlab", building_storeys=ifc_building_storeys
)

# 2. Simplify the footprint
tol = 10  # tolerance for simplification
footprint_simpl = footprint_ifc.simplify(tol, preserve_topology=True)

# 3. Detect features on simplified and original footprints
angle_thresh = 30
features_simpl = detect_features(footprint_simpl, angle_threshold_deg=angle_thresh)
features_orig = detect_features(footprint_ifc, angle_threshold_deg=angle_thresh)

# 4. Filter the original features
min_edge_len = 15.0
filtered_dist = filter_features_by_feature_edge_length(
    features_orig, min_edge_len=min_edge_len
)
min_area = 15
filtered_tri = filter_features_by_feature_triangle_area(
    features_orig, min_area=min_area
)

# 5. Plot side by side
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Simplified footprint & its detected corners
ax = axes[0]
if not footprint_ifc.is_empty:
    for poly in footprint_ifc.geoms:
        x, y = poly.exterior.xy
        ax.plot(x, y, color="blue", alpha=0.5)

# overlay original detected features
if features_orig.size:
    ax.scatter(
        features_orig[:, 2],
        features_orig[:, 3],
        color="mistyrose",
        s=50,
        label=f"Detected (original): {len(features_orig)}",
    )

# detected on simplified
if features_simpl.size:
    ax.scatter(
        features_simpl[:, 2],
        features_simpl[:, 3],
        color="red",
        s=50,
        label=f"Detected (simplified): {len(features_simpl)}",
    )

ax.set_title(f"Shapely Simplified (tol={tol} - Douglas-Peucker)")
ax.set_aspect("equal", "box")
ax.legend()

# Panel 2: Original footprint, filter by edge‐length
ax = axes[1]
if not footprint_ifc.is_empty:
    for poly in footprint_ifc.geoms:
        x, y = poly.exterior.xy
        ax.plot(x, y, color="blue", alpha=0.5)
# all detected
if features_orig.size:
    ax.scatter(
        features_orig[:, 2],
        features_orig[:, 3],
        color="mistyrose",
        label=f"All Detected: {len(features_orig)}",
    )
# filtered
if filtered_dist.size:
    ax.scatter(
        filtered_dist[:, 2],
        filtered_dist[:, 3],
        color="red",
        label=f"Kept (edge ≥ {min_edge_len}): {len(filtered_dist)}",
    )
ax.set_title(f"Filter by Feature-Edge Length ≥ {min_edge_len}")
ax.set_aspect("equal", "box")
ax.legend()

# Panel 3: Original footprint, filter by feature‐triangle area
ax = axes[2]
if not footprint_ifc.is_empty:
    for poly in footprint_ifc.geoms:
        x, y = poly.exterior.xy
        ax.plot(x, y, color="blue", alpha=0.5)
# all detected
if features_orig.size:
    ax.scatter(
        features_orig[:, 2],
        features_orig[:, 3],
        color="mistyrose",
        label=f"All Detected: {len(features_orig)}",
    )
# filtered
if filtered_tri.size:
    ax.scatter(
        filtered_tri[:, 2],
        filtered_tri[:, 3],
        color="red",
        label=f"Kept (tri area ≥ {min_area}): {len(filtered_tri)}",
    )
ax.set_title(f"Filter by Feature‐Triangle Area ≥ {min_area}")
ax.set_aspect("equal", "box")
ax.legend()

plt.tight_layout()
plt.show()