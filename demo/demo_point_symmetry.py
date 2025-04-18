from source.transformation_horizontal.handle_point_symmetry import check_point_symmetry
from source.transformation_horizontal.estimate_rigid_transformation import estimate_rigid_transformation, refine_rigid_transformation
from source.transformation_horizontal.detect_features import detect_features, filter_features_by_feature_triangle_area
from source.transformation_horizontal.rigid_transformation import Rigid_Transformation

from shapely.geometry import Polygon, MultiPolygon
import numpy as np
from shapely.affinity import translate, rotate
import matplotlib.pyplot as plt

# Create source and target polygons
source = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
# Create a target polygon that is translated and rotated relative to the source

# First translate the source polygon
target = translate(source, xoff=4, yoff=2)  
# Then rotate it by 30 degrees around its centroid
target = rotate(target, angle=30, origin='centroid')

# Convert to MultiPolygon as required by the feature detection
source_mp = MultiPolygon([source])
target_mp = MultiPolygon([target])

# Detect features on both polygons
source_features = detect_features(source_mp, angle_threshold_deg=30)
target_features = detect_features(target_mp, angle_threshold_deg=30)

# Filter features
source_features_filtered = filter_features_by_feature_triangle_area(source_features, min_area=0.1)
target_features_filtered = filter_features_by_feature_triangle_area(target_features, min_area=0.1)

print(f"Source features: {len(source_features_filtered)}")
print(f"Target features: {len(target_features_filtered)}")

# Find the index of the corner at (0,0) in source features
source_corner_idx = None
for i, feature in enumerate(source_features_filtered):
    x, y = feature[2:4]
    if np.isclose(x, 0) and np.isclose(y, 0):
        source_corner_idx = i
        break

if source_corner_idx is None:
    print("Error: Could not find the (0,0) corner in source features.")
    exit(1)

# Find the corresponding corner in target features
# For demo purposes, let's assume it's the first feature
# In a real application, you would need to determine the correct correspondence
target_corner_idx = 0  # This is an assumption, would need specific logic in real use

# Run estimation with restricted mode using the specified feature correspondence
trans, inliers = estimate_rigid_transformation(
    source_features_filtered, 
    target_features_filtered,
    distance_tol=0.5,
    angle_tol_deg=10,
    restricted=True,
    fixed_source_idx=source_corner_idx,
    fixed_target_idx=target_corner_idx
)

if trans is None:
    print("Failed to estimate transformation.")
    exit(1)

# Refine the transformation if needed
refined_trans = refine_rigid_transformation(inliers) if inliers else None

print(f"Estimated transformation: {trans}")
print(f"Inliers: {len(inliers)}")
if refined_trans:
    print(f"Refined transformation: {refined_trans}")

# Apply transformation to source to verify
transformed_source = trans.transform(source)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Source with feature indices
axes[0].plot(*source.exterior.xy, 'b-', label='Source')
for i, f in enumerate(source_features_filtered):
    axes[0].scatter(f[2], f[3], c='blue', marker='o')
    axes[0].annotate(f"{i}", (f[2]+0.05, f[3]+0.05), fontsize=9)
axes[0].set_aspect('equal')
axes[0].grid(True)
axes[0].set_title("Source with Feature Indices")
axes[0].legend()

# Plot 2: Target with feature indices
axes[1].plot(*target.exterior.xy, 'g-', label='Target')
for i, f in enumerate(target_features_filtered):
    axes[1].scatter(f[2], f[3], c='green', marker='x')
    axes[1].annotate(f"{i}", (f[2]+0.05, f[3]+0.05), fontsize=9)
axes[1].set_aspect('equal')
axes[1].grid(True)
axes[1].set_title("Target with Feature Indices")
axes[1].legend()

# Plot 3: Combined view with source, target, and transformed source
axes[2].plot(*source.exterior.xy, 'b--', label='Source')
axes[2].plot(*target.exterior.xy, 'g-', label='Target')
axes[2].plot(*transformed_source.exterior.xy, 'r:', label='Transformed Source')

# Add features with indices to the combined view
for i, f in enumerate(source_features_filtered):
    axes[2].scatter(f[2], f[3], c='blue', marker='o')
    axes[2].annotate(f"{i}", (f[2]+0.05, f[3]+0.05), fontsize=9, color='blue')
for i, f in enumerate(target_features_filtered):
    axes[2].scatter(f[2], f[3], c='green', marker='x')
    axes[2].annotate(f"{i}", (f[2]+0.05, f[3]+0.05), fontsize=9, color='green')

axes[2].set_aspect('equal')
axes[2].grid(True)
axes[2].legend()
axes[2].set_title("Combined View with Feature Correspondence")

# Highlight the fixed correspondence
if source_corner_idx is not None and target_corner_idx is not None:
    src_fixed = source_features_filtered[source_corner_idx]
    tgt_fixed = target_features_filtered[target_corner_idx]
    axes[2].scatter(src_fixed[2], src_fixed[3], c='red', s=100, alpha=0.5, marker='o')
    axes[2].scatter(tgt_fixed[2], tgt_fixed[3], c='red', s=100, alpha=0.5, marker='x')
    axes[2].plot([src_fixed[2], tgt_fixed[2]], [src_fixed[3], tgt_fixed[3]], 'r--', alpha=0.5)

plt.tight_layout()
plt.suptitle("Restricted Rigid Registration with Fixed Feature Correspondence", y=1.05)
plt.show()

# Print the fixed correspondence details
print(f"Fixed correspondence: Source feature {source_corner_idx} to Target feature {target_corner_idx}")

