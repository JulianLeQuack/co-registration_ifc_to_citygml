import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from source.transformation_horizontal.create_footprints.create_CityGML_footprint import create_CityGML_footprint
from source.transformation_horizontal.create_footprints.create_DXF_footprint_polygon import create_DXF_footprint_polygon
from source.transformation_horizontal.create_footprints.create_IFC_footprint_polygon import create_IFC_footprint_polygon


def compute_turning_angles(points: np.array) -> np.array:
    """
    Compute the turning angles (in radians) at each vertex of a closed polygon.
    'points' is assumed to be a (n,2) numpy array of 2D vertices.
    """
    n = len(points)
    angles = []
    for i in range(n):
        prev = points[i - 1]
        curr = points[i]
        nxt = points[(i + 1) % n]
        # Compute vectors for the incoming and outgoing edges.
        v1 = curr - prev
        v2 = nxt - curr
        # Calculate the signed angle.
        angle = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
        angles.append(angle)
    return np.array(angles)

def detect_features(footprint, angle_threshold_deg=30) -> np.array:
    """
    Detect features (corners) in a MultiPolygon based on turning angles.
    
    For each polygon in the MultiPolygon, the exterior coordinates (excluding the duplicate closing point)
    are used to compute turning angles at each vertex. A vertex is considered a feature if the absolute 
    turning angle (in degrees) exceeds the given threshold.
    
    Returns:
        A numpy array where each row represents a detected feature in the form:
        [polygon_index, vertex_index, x_coordinate, y_coordinate, turning_angle_deg]
    """
    features = []
    for poly_idx, poly in enumerate(footprint.geoms):
        # Remove the duplicate last point.
        pts = np.array(poly.exterior.coords[:-1])
        angles = compute_turning_angles(pts)
        angles_deg = np.degrees(angles)
        for i, angle in enumerate(angles_deg):
            if abs(angle) >= angle_threshold_deg:
                # Record: polygon index, vertex index, x, y, turning angle.
                features.append([poly_idx, i, pts[i, 0], pts[i, 1], angle])
    return np.array(features).reshape(-1, 5) if features else np.empty((0, 5))


def filter_features_by_edge_length(features: np.array, footprint, min_edge_len=2.0) -> np.array:
    """
    Filters out detected features if both the incoming and outgoing edge lengths are below min_edge_len.
    """
    filtered_features = []
    for feature in features:
        poly_idx, vertex_idx, x, y, angle = feature
        poly = footprint.geoms[int(poly_idx)]
        pts = np.array(poly.exterior.coords[:-1])
        n = len(pts)
        idx = int(vertex_idx) % n
        prev_idx = (idx - 1) % n
        next_idx = (idx + 1) % n
        edge_in = np.linalg.norm(pts[idx] - pts[prev_idx])
        edge_out = np.linalg.norm(pts[next_idx] - pts[idx])
        if edge_in >= min_edge_len or edge_out >= min_edge_len:
            filtered_features.append(feature)
    return np.array(filtered_features).reshape(-1, 5) if filtered_features else np.empty((0, 5))


def filter_features_by_triangle_area(features: np.array, footprint, min_area=15) -> np.array:
    """
    Filters out detected features based on the area of the triangle
    formed by the feature vertex and its two neighbors (using all vertices).
    """
    filtered_features = []
    for feature in features:
        poly_idx, vertex_idx, x, y, angle = feature
        poly = footprint.geoms[int(poly_idx)]
        pts = np.array(poly.exterior.coords[:-1])
        n = len(pts)
        idx = int(vertex_idx) % n
        prev_idx = (idx - 1) % n
        next_idx = (idx + 1) % n
        area = 0.5 * abs((pts[prev_idx][0] * (pts[idx][1] - pts[next_idx][1]) +
                          pts[idx][0] * (pts[next_idx][1] - pts[prev_idx][1]) +
                          pts[next_idx][0] * (pts[prev_idx][1] - pts[idx][1])))
        if area >= min_area:
            filtered_features.append(feature)
    return np.array(filtered_features).reshape(-1, 5) if filtered_features else np.empty((0, 5))


def filter_features_by_feature_triangle_area(features: np.array, min_area=0.5) -> np.array:
    """
    Filters detected features based on the area of triangles formed between adjacent detected features.
    This function uses only the features (the detected corners), not all original polygon vertices.
    
    Args:
        features: Array of detected features, each in the form 
                  [polygon_index, vertex_index, x_coordinate, y_coordinate, turning_angle_deg]
        min_area: Minimum triangle area threshold. Only features whose spanned triangle area
                  (with its adjacent features in the sorted order) is at least min_area are retained.
    
    Returns:
        A numpy array of filtered features (in the same 5-column format) or an empty array if none pass.
    """
    if features.size == 0:
        return np.empty((0, 5))
    
    # Group features by polygon.
    grouped_features = group_features_by_polygon(features)
    filtered_features = []
    
    # For each feature, compute the triangle area using its adjacent features in the grouped order.
    for feature in features:
        result = compute_triangle_area_from_features(feature, grouped_features)
        if result is not None:
            area, _, _, _ = result
            if area >= min_area:
                filtered_features.append(feature)
    
    return np.array(filtered_features).reshape(-1, 5) if filtered_features else np.empty((0, 5))


def group_features_by_polygon(features: np.array) -> dict:
    """
    Build a dictionary mapping each polygon index to a list of its detected features,
    sorted by the feature’s vertex index.
    Each feature is assumed to be in the form:
       [polygon_index, vertex_index, x, y, turning_angle_deg]
    """
    groups = {}
    for f in features:
        poly_idx = int(f[0])
        if poly_idx not in groups:
            groups[poly_idx] = []
        groups[poly_idx].append(f)
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda f: int(f[1]))
    return groups

def compute_triangle_area_from_features(feature, grouped_features):
    """
    Given a feature and the grouped features (only features detected),
    compute the area of the triangle formed by the previous, current, 
    and next detected features (cyclic within that polygon).
    
    Returns:
        A tuple (area, p, c, npnt) where:
          - area is the computed triangle area,
          - p, c, npnt are numpy arrays of the coordinates of the previous, current,
            and next features respectively.
        Returns None if there are fewer than 3 features in the group.
    """
    poly_idx = int(feature[0])
    group = grouped_features.get(poly_idx, [])
    if len(group) < 3:
        return None
    f_idx = None
    for i, f in enumerate(group):
        if int(f[1]) == int(feature[1]):
            f_idx = i
            break
    if f_idx is None:
        return None
    n = len(group)
    prev_f = group[(f_idx - 1) % n]
    next_f = group[(f_idx + 1) % n]
    p = np.array([prev_f[2], prev_f[3]])
    c = np.array([feature[2], feature[3]])
    npnt = np.array([next_f[2], next_f[3]])
    # Compute area using the standard triangle area formula.
    area = 0.5 * abs(p[0]*(c[1] - npnt[1]) + c[0]*(npnt[1] - p[1]) + npnt[0]*(p[1] - c[1]))
    return area, p, c, npnt


def plot_features(ax, footprint, detected_features, filtered_features, title, grouped_features=None):
    # Plot each polygon.
    if not footprint.is_empty:
        for poly in footprint.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='blue', alpha=0.5)
    
    # Plot all detected features.
    if detected_features.size and detected_features.ndim == 2:
        ax.scatter(detected_features[:, 2], detected_features[:, 3],
                   color='mistyrose', label=f'All Detected: {len(detected_features)}')
        for idx, feature in enumerate(detected_features):
            ax.text(feature[2] + 0.2, feature[3] + 0.2, str(idx),
                    fontsize=8, color='green', weight='bold')
    
    # Plot filtered features.
    if filtered_features.size and filtered_features.ndim == 2:
        ax.scatter(filtered_features[:, 2], filtered_features[:, 3],
                   color='red', label=f'Filtered: {len(filtered_features)}')
    
    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")


if __name__ == "__main__":
    # Load footprints.
    ifc_path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"
    footprint_ifc = create_IFC_footprint_polygon(ifc_path, ifc_type="IfcSlab")

    dxf_path = "./test_data/dxf/01-05-0507_EG.1.dxf"
    footprint_dxf = create_DXF_footprint_polygon(
        dxf_path,
        layer_name="A_09_TRAGDECKE",
        use_origin_filter=False,
        origin_threshold=10.0
    )

    citygml_path = "./test_data/citygml/TUM_LoD2_Full_withSurrounds.gml"
    citygml_buildings = ["DEBY_LOD2_4959457"]
    footprint_citygml = create_CityGML_footprint(citygml_path, citygml_buildings)
    
    # Detect features.
    features_ifc = detect_features(footprint_ifc, angle_threshold_deg=30)
    features_dxf = detect_features(footprint_dxf, angle_threshold_deg=30)
    features_citygml = detect_features(footprint_citygml, angle_threshold_deg=30)
    
    # Filter features by edge length.
    min_edge_len = 5
    filtered_features_ifc = filter_features_by_edge_length(features_ifc, footprint_ifc, min_edge_len=min_edge_len)
    filtered_features_dxf = filter_features_by_edge_length(features_dxf, footprint_dxf, min_edge_len=min_edge_len)
    filtered_features_citygml = filter_features_by_edge_length(features_citygml, footprint_citygml, min_edge_len=min_edge_len)
    
    # Filter features by triangle area using features (not vertices).
    min_area = 15
    filtered_features_ifc_area = filter_features_by_feature_triangle_area(features_ifc, min_area=min_area)
    filtered_features_dxf_area = filter_features_by_feature_triangle_area(features_dxf, min_area=min_area)
    filtered_features_citygml_area = filter_features_by_feature_triangle_area(features_citygml, min_area=min_area)
    
    # Filter features by feature triangle area.
    filtered_features_ifc_feature_area = filter_features_by_feature_triangle_area(features_ifc, min_area=min_area)
    filtered_features_dxf_feature_area = filter_features_by_feature_triangle_area(features_dxf, min_area=min_area)
    filtered_features_citygml_feature_area = filter_features_by_feature_triangle_area(features_citygml, min_area=min_area)
    
    # Group features for triangle area computation (using all detected features).
    grouped_features_ifc = group_features_by_polygon(features_ifc)
    grouped_features_dxf = group_features_by_polygon(features_dxf)
    grouped_features_citygml = group_features_by_polygon(features_citygml)
    
    # Create a 2x3 figure.
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    
    # Top row: Edge Length filtered features.
    plot_features(axes[0, 0], footprint_ifc, features_ifc, filtered_features_ifc, "IFC: Filter by Edge Length")
    plot_features(axes[0, 1], footprint_dxf, features_dxf, filtered_features_dxf, "DXF: Filter by Edge Length")
    plot_features(axes[0, 2], footprint_citygml, features_citygml, filtered_features_citygml, "CityGML: Filter by Edge Length")
    
    # Bottom row: Triangle Area plots showing all spanned triangles from detected features.
    plot_features(axes[1, 0], footprint_ifc, features_ifc, filtered_features_ifc_area,
                  "IFC: Triangle Areas (All Triangles)", grouped_features_ifc)
    plot_features(axes[1, 1], footprint_dxf, features_dxf, filtered_features_dxf_area,
                  "DXF: Triangle Areas (All Triangles)", grouped_features_dxf)
    plot_features(axes[1, 2], footprint_citygml, features_citygml, filtered_features_citygml_area,
                  "CityGML: Triangle Areas (All Triangles)", grouped_features_citygml)
    
    plt.tight_layout()
    plt.show()


# fig, (ax1, ax2) = plt.subplots(2,1, figsize=(24, 12))

# # Left: All detected corners
# for poly in footprint_ifc.geoms:
#     x, y = poly.exterior.xy
#     ax1.plot(x, y, color='blue', alpha=0.5)
# if features_ifc.size:
#     ax1.scatter(features_ifc[:, 2], features_ifc[:, 3], color='red', s=80, label=f'All Detected: {len(features_ifc)}')
#     # Place label in the middle of the plot
#     mid_x = (ax1.get_xlim()[0] + ax1.get_xlim()[1]) / 2
#     mid_y = (ax1.get_ylim()[0] + ax1.get_ylim()[1]) / 2
#     ax1.text(mid_x, mid_y, f'All Detected: {len(features_ifc)}', fontsize=22, color='black', ha='center', va='center', weight='bold', alpha=1)
# ax1.set_title("IFC Footprint - All Detected Corners")
# ax1.set_aspect("equal", "box")

# # Right: Filtered corners by feature triangle area
# for poly in footprint_ifc.geoms:
#     x, y = poly.exterior.xy
#     ax2.plot(x, y, color='blue', alpha=0.5)
# if filtered_features_ifc_feature_area.size:
#     ax2.scatter(filtered_features_ifc_feature_area[:, 2], filtered_features_ifc_feature_area[:, 3], color='red', s=80, label=f'After Filtering: {len(filtered_features_ifc_feature_area)}')
#     # Place label in the middle of the plot
#     mid_x = (ax2.get_xlim()[0] + ax2.get_xlim()[1]) / 2
#     mid_y = (ax2.get_ylim()[0] + ax2.get_ylim()[1]) / 2
#     ax2.text(mid_x, mid_y, f'After Filtering: {len(filtered_features_ifc_feature_area)}', fontsize=22, color='black', ha='center', va='center', weight='bold', alpha=1)
# ax2.set_title("IFC Footprint - Filtered Corners (Feature Triangle Area)")
# ax2.set_aspect("equal", "box")

# plt.tight_layout()
# plt.show()