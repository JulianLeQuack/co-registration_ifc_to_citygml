import numpy as np
import matplotlib.pyplot as plt

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
    The rationale is that a 'corner' formed by very short edges is likely a minor recession detail.
    
    Parameters:
        features (np.array): Array with shape (n,5) as produced by detect_features.
        footprint: A MultiPolygon containing the original footprints.
        min_edge_len (float): Minimum edge length (in same units as footprint) to consider a corner relevant.
    
    Returns:
        np.array: Filtered features in the same shape, only including features where at least one edge length
                  (incoming or outgoing) is greater than or equal to min_edge_len.
    """
    filtered_features = []
    for feature in features:
        poly_idx, vertex_idx, x, y, angle = feature
        poly = footprint.geoms[int(poly_idx)]
        pts = np.array(poly.exterior.coords[:-1])  # use the same vertices used in detect_features
        n = len(pts)
        # Ensure the vertex index is within bounds
        idx = int(vertex_idx) % n
        prev_idx = (idx - 1) % n
        next_idx = (idx + 1) % n
        edge_in = np.linalg.norm(pts[idx] - pts[prev_idx])
        edge_out = np.linalg.norm(pts[next_idx] - pts[idx])
        # Retain the feature if at least one of the edges is longer than min_edge_len.
        if edge_in >= min_edge_len or edge_out >= min_edge_len:
            filtered_features.append(feature)
    return np.array(filtered_features).reshape(-1, 5) if filtered_features else np.empty((0, 5))


if __name__ == "__main__":
    # Either use a CityGML footprint or an IFC footprint. In this example, we use IFC.
    # ifc_path = "./test_data/ifc/3.003 01-05-0507_EG.ifc"
    ifc_path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"
    footprint_ifc = create_IFC_footprint_polygon(ifc_path, ifc_type="IfcSlab")

    dxf_path = "./test_data/dxf/01-05-0501_EG.dxf"
    footprint_dxf = create_DXF_footprint_polygon(
        dxf_path,
        layer_name="A_09_TRAGDECKE",
        use_origin_filter=False,
        origin_threshold=10.0
    )

    citygml_path = "./test_data/citygml/TUM_LoD2_Full_withSurrounds.gml"
    citygml_buildings = ["DEBY_LOD2_4959457"]
    footprint_citygml = create_CityGML_footprint(
        citygml_path,
        citygml_buildings,
    )
    
    # First, detect features based on turning angle.
    features_ifc = detect_features(footprint_ifc, angle_threshold_deg=30)
    features_dxf = detect_features(footprint_dxf, angle_threshold_deg=30)
    features_citygml = detect_features(footprint_citygml, angle_threshold_deg=30)
    
    # Now apply edge length filtering. For example, we consider features with both adjacent
    # edges shorter than 2.0 to be minor and filter them out.
    min_edge_len = 7  # adjust as needed (e.g., in meters)
    filtered_features_ifc = filter_features_by_edge_length(features_ifc, footprint_ifc, min_edge_len=min_edge_len)
    filtered_features_dxf = filter_features_by_edge_length(features_dxf, footprint_dxf, min_edge_len=min_edge_len)
    filtered_features_citygml = filter_features_by_edge_length(features_citygml, footprint_citygml, min_edge_len=min_edge_len)

    print(f"Detected Features in IFC: {features_ifc[0]}")
    print(f"Filtered Features in IFC: {filtered_features_ifc[0]}")
    print(f"Detected Features in DXF: {features_dxf[0]}")
    print(f"Filtered Features in DXF: {filtered_features_dxf[0]}")
    print(f"Detected Features in CityGML: {features_citygml[0]}")
    print(f"Filtered Features in CityGML: {filtered_features_citygml[0]}")
    
    # Create a figure with 3 subplots, one for each footprint type.
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    # Define a function to plot a footprint and features on an axis.
    def plot_features(ax, footprint, detected_features, filtered_features, title):
        # Plot each polygon in the footprint.
        if not footprint.is_empty:
            for poly in footprint.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color='blue', alpha=0.5)
        
        # Plot all detected features as light red.
        if detected_features.size and detected_features.ndim == 2:
            ax.scatter(detected_features[:, 2], detected_features[:, 3], 
                       color='mistyrose', label=f'All Detected: {len(detected_features)}')
        # Plot filtered features as strong red.
        if filtered_features.size and filtered_features.ndim == 2:
            ax.scatter(filtered_features[:, 2], filtered_features[:, 3], 
                       color='red', label=f'Filtered: {len(filtered_features)}')
        else:
            ax.text(0.5, 0.5, "No robust features", transform=ax.transAxes, ha="center")
            
        ax.set_title(title)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend()
        ax.set_aspect("equal", adjustable="box")
    
    # Plot for IFC footprint.
    plot_features(axes[0], footprint_ifc, features_ifc, filtered_features_ifc, "IFC Footprint Features")
    
    # Plot for DXF footprint.
    plot_features(axes[1], footprint_dxf, features_dxf, filtered_features_dxf, "DXF Footprint Features")
    
    # Plot for CityGML footprint.
    plot_features(axes[2], footprint_citygml, features_citygml, filtered_features_citygml, "CityGML Footprint Features")
    
    plt.tight_layout()
    plt.show()
