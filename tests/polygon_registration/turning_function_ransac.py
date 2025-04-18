import numpy as np
import matplotlib.pyplot as plt
import random

from source.transformation_horizontal.create_footprints.create_CityGML_footprint import *
from source.transformation_horizontal.create_footprints.create_IFC_footprint import *
#from source.transformation_horizontal.create_footprints.create_hull import create_concave_hull

def compute_turning_angles(points):
    """
    Compute the turning angles (in radians) at each vertex of a closed polygon.
    For each vertex, the angle is computed between the incoming and outgoing edges.
    """
    angles = []
    n = len(points)
    for i in range(n):
        prev = points[i - 1]
        curr = points[i]
        nxt = points[(i + 1) % n]
        # Vectors for incoming and outgoing edges
        v1 = curr - prev
        v2 = nxt - curr
        # Signed angle using arctan2 of the determinant and dot product.
        angle = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
        angles.append(angle)
    return np.array(angles)

def detect_features(points, angle_threshold_deg=30):
    """
    Detect features based on the turning angles.
    
    A vertex is marked as a feature (corner) if the absolute turning angle 
    (in degrees) is above a given threshold.
    
    Returns a list of tuples: (vertex index, point coordinate, turning angle in degrees).
    """
    angles = compute_turning_angles(points)
    angles_deg = np.degrees(angles)
    features = []
    for i, angle in enumerate(angles_deg):
        if abs(angle) >= angle_threshold_deg:
            features.append((i, points[i], angle))
    return features

def estimate_transformation_from_2pairs(p1, p2, q1, q2):
    """
    Estimate the 2D rigid transformation (rotation and translation) from two pairs of corresponding points.
    
    p1, p2: Two points from polygon1.
    q1, q2: The corresponding points from polygon2.
    """
    # Compute vectors between the pairs.
    v1 = p2 - p1
    v2 = q2 - q1
    # Compute the angle of each vector.
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    # The rotation required is the difference between these angles.
    theta = angle2 - angle1
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    # Translation: align the first point.
    t = q1 - R @ p1
    return R, t

def apply_transform(points, R, t):
    """
    Apply a rigid transformation (rotation R and translation t) to a set of points.
    """
    return (R @ points.T).T + t

def ransac_rigid_transform(features1, features2, iterations=1000, distance_tol=5.0, angle_tol_deg=10.0):
    """
    Use RANSAC to estimate the best rigid transformation that aligns features1 to features2.
    
    Parameters:
      features1, features2: Lists of tuples (index, point, angle in degrees) from each polygon.
      iterations: Number of RANSAC iterations.
      distance_tol: Maximum allowed distance (in same units as points) for a transformed feature to be considered an inlier.
      angle_tol_deg: Maximum allowed difference in turning angle (degrees) between a transformed feature and a candidate match.
      
    Returns:
      best_R: The best rotation matrix.
      best_t: The best translation vector.
      best_inliers: List of inlier features from polygon1 that agree with the transformation.
    """
    best_inlier_count = 0
    best_R = None
    best_t = None
    best_inliers = []
    angle_tol = np.radians(angle_tol_deg)
    
    if len(features1) < 2 or len(features2) < 2:
        print("Not enough features for RANSAC.")
        return None, None, []
    
    for i in range(iterations):
        # Randomly pick two features from polygon1.
        sample_indices = random.sample(range(len(features1)), 2)
        f1_1 = features1[sample_indices[0]]
        f1_2 = features1[sample_indices[1]]
        p1, p2 = f1_1[1], f1_2[1]
        a1, a2 = np.radians(f1_1[2]), np.radians(f1_2[2])
        
        # For each feature, find candidate matches in polygon2 based on turning angle similarity.
        candidates1 = [f for f in features2 if abs(np.radians(f[2]) - a1) < angle_tol]
        candidates2 = [f for f in features2 if abs(np.radians(f[2]) - a2) < angle_tol]
        
        if not candidates1 or not candidates2:
            continue
        
        # Randomly choose one candidate for each.
        f2_1 = random.choice(candidates1)
        f2_2 = random.choice(candidates2)
        q1, q2 = f2_1[1], f2_2[1]
        
        # Avoid degenerate cases.
        if np.linalg.norm(p2 - p1) < 1e-3 or np.linalg.norm(q2 - q1) < 1e-3:
            continue
        
        # Estimate the transformation from the two pairs.
        R, t = estimate_transformation_from_2pairs(p1, p2, q1, q2)
        
        # Evaluate inliers: transform all features from polygon1 and check for a nearby feature in polygon2.
        inliers = []
        for f in features1:
            idx, pt, ang = f
            pt_trans = R @ pt + t
            matched = False
            for g in features2:
                idx2, pt2, ang2 = g
                if np.linalg.norm(pt_trans - pt2) < distance_tol and abs(np.radians(ang) - np.radians(ang2)) < angle_tol:
                    matched = True
                    break
            if matched:
                inliers.append(f)
        
        # Keep the transformation if it has more inliers than previous iterations.
        if len(inliers) > best_inlier_count:
            best_inlier_count = len(inliers)
            best_R = R
            best_t = t
            best_inliers = inliers
            
    return best_R, best_t, best_inliers

def main():

    ifc_path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"
    citygml_path = "./test_data/citygml/DEBY_LOD2_4959457.gml"

    # Define two example polygons.

    polygon1 = create_concave_hull(create_IFC_footprint(ifc_path), 0.1) 
    polygon2 = create_CityGML_footprint(citygml_path)

    # polygon1 = np.genfromtxt("./test_data/points/floorplan-vertices-automatic-sparse.txt", dtype=np.double, delimiter=",", encoding="utf-8-sig")
    # polygon2 = np.genfromtxt("./test_data/points/footprint-vertices-automatic-sparse.txt", dtype=np.double, delimiter=",", encoding="utf-8-sig")


    # Detect features (corners) using the turning function.
    features1 = detect_features(polygon1, angle_threshold_deg=70)
    features2 = detect_features(polygon2, angle_threshold_deg=70)

    print(f"Polygon1: {len(features1)} features")
    print(f"Polygon2: {len(features2)} features")
    
    # print("Polygon1 features (index, point, angle):")
    # for f in features1:
    #     print(f)
    # print("\nPolygon2 features (index, point, angle):")
    # for f in features2:
    #     print(f)
    
    # Use RANSAC to find the best rigid transformation aligning polygon1's features to polygon2's features.
    R, t, inliers = ransac_rigid_transform(features1, features2,
                                             iterations=len(features1) * len(features2),
                                             distance_tol=10,
                                             angle_tol_deg=30)
    
    if R is None:
        print("RANSAC failed to find a valid transformation.")
        return
    
    print("\nBest transformation found:")
    print("Rotation matrix R:\n", R)
    print("Translation vector t:\n", t)
    print("Number of inliers:", len(inliers))
    
    # Apply the transformation to polygon1.
    polygon1_transformed = apply_transform(polygon1, R, t)
    
    # Visualize the results.
    plt.figure(figsize=(8, 6))
    # plt.plot(polygon1[:, 0], polygon1[:, 1], 'bo-', label='Polygon1 (Original)')
    plt.plot(polygon1_transformed[:, 0], polygon1_transformed[:, 1], 'ro-', label='Polygon1 (Transformed)')
    plt.plot(polygon2[:, 0], polygon2[:, 1], 'go-', label='Polygon2')
    plt.axis('equal')
    plt.title("RANSAC Alignment using Turning Function Features")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
