import numpy as np
import matplotlib.pyplot as plt

from source.transformation_horizontal.create_footprints.create_CityGML_footprint import create_CityGML_footprint
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
    ifc_path = "./test_data/ifc/3.003 01-05-0507_EG.ifc"
    # ifc_path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"
    footprint = create_IFC_footprint_polygon(ifc_path, ifc_type="IfcSlab")
    
    if footprint is None:
        print("Failed to create footprint from the IFC file.")
        exit(1)
    
    # First, detect features based on turning angle.
    features = detect_features(footprint, angle_threshold_deg=30)
    
    # Now apply edge length filtering. For example, we consider features with both adjacent
    # edges shorter than 2.0 to be minor and filter them out.
    min_edge_len = 7  # adjust as needed (e.g., in meters)
    filtered_features = filter_features_by_edge_length(features, footprint, min_edge_len=min_edge_len)
    
    # Plot the footprint and the detected (and filtered) features.
    plt.figure(figsize=(10, 10))
    if not footprint.is_empty:
        for poly in footprint.geoms:
            x, y = poly.exterior.xy
            plt.plot(x, y, color='blue', alpha=0.5)
            
    # Plot original detected features (if any) in a light red color.
    if features.size and features.ndim == 2:
        plt.scatter(features[:, 2], features[:, 3], color='mistyrose', label=f'All Detected Features: {len(features)}')
    
    # Plot filtered robust features in a stronger red.
    if filtered_features.size and filtered_features.ndim == 2:
        plt.scatter(filtered_features[:, 2], filtered_features[:, 3], color='red', label=f'Filtered Features: {len(filtered_features)}')
    else:
        print("No features passed the edge length filtering.")
    
    plt.title("Detected Features with Edge Length Filtering")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.axis('equal')
    plt.show()
