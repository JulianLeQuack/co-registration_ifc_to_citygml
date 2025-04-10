import numpy as np
import matplotlib.pyplot as plt

from source.transformation_horizontal.create_footprints.create_CityGML_footprint import create_CityGML_footprint

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
        # Compute vectors for incoming and outgoing edges.
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


if __name__ == "__main__":
    # Adjust the file path as needed (remove the extra dot before test_data)
    citygml_path = "./test_data/citygml/TUM_LoD2_Full_withSurrounds.gml"
    citygml_building = ["DEBY_LOD2_4959457"]
    footprint = create_CityGML_footprint(citygml_path, citygml_building)

    features = detect_features(footprint, angle_threshold_deg=30)
    print(f"Detected features: {features}")

    plt.figure(figsize=(10, 10))
    if not footprint.is_empty:
        for poly in footprint.geoms:
            x, y = poly.exterior.xy
            plt.plot(x, y, color='blue', alpha=0.5)
            # plt.scatter(x, y, color='blue', alpha=0.5)
    # Only plot features if they were detected.
    if features.size and features.ndim == 2:
        plt.scatter(features[:, 2], features[:, 3], color='red', label=f'Detected Features: {len(features)}')
    else:
        print("No features detected to plot.")
    plt.title("Detected Features in CityGML Footprint")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.axis('equal')
    plt.show()