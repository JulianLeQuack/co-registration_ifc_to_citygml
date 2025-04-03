import numpy as np
import matplotlib.pyplot as plt
import itertools

from source.transformation_horizontal.create_footprints.create_CityGML_footprint import *
from source.transformation_horizontal.create_footprints.create_IFC_footprint import *
from source.transformation_horizontal.create_footprints.create_hull import create_concave_hull

from source.transformation_horizontal.rigid_transformation import Rigid_Transformation

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

def estimate_transformation_from_2pairs(source1, source2, target1, target2) -> Rigid_Transformation:
    """
    Estimate the 2D rigid transformation (rotation and translation) from two pairs of corresponding points.

    Parameters:
        source1, source2: Two 2D points from the source shape (e.g., polygon).
        target1, target2: Corresponding 2D points in the target shape.

    Returns:
        A new Rigid_Transformation object representing the estimated transformation.
    """
    # Compute direction vectors.
    vector1 = source2 - source1
    vector2 = target2 - target1

    # Compute angles of those vectors.
    angle1 = np.arctan2(vector1[1], vector1[0])
    angle2 = np.arctan2(vector2[1], vector2[0])

    # Compute the angle difference.
    theta = angle2 - angle1

    # Create rotation matrix.
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Compute translation.
    t = target1 - R @ source1

    # Return new transformation object.
    return Rigid_Transformation(t=t, theta=theta)


def ransac_rigid_transform(features1, features2, distance_tol=5.0, angle_tol_deg=10.0):
    """
    Use RANSAC to estimate the best rigid transformation that aligns features1 to features2.
    
    Parameters:
      features1, features2: Lists of tuples (index, point, angle in degrees) from each polygon.
      distance_tol: Maximum allowed distance for a transformed feature to be considered an inlier.
      angle_tol_deg: Maximum allowed difference in turning angle (degrees) for a candidate match.
      
    Returns:
      best_transformation: A Rigid_Transformation object representing the best transformation.
      best_inliers: List of inlier features from polygon1 that agree with the transformation.
    """
    import itertools
    
    best_inlier_count = 0
    best_transformation = None
    best_inliers = []
    angle_tol = np.radians(angle_tol_deg)
    
    if len(features1) < 2 or len(features2) < 2:
        print("Not enough features for RANSAC.")
        return None, []
    
    # Loop over all pairs of features from the first set.
    for f1_1, f1_2 in itertools.combinations(features1, 2):
        p1, p2 = f1_1[1], f1_2[1]
        a1, a2 = np.radians(f1_1[2]), np.radians(f1_2[2])
        
        # Find candidate matches in features2 for each feature based on similar turning angles.
        candidates1 = [f for f in features2 if abs(np.radians(f[2]) - a1) < angle_tol]
        candidates2 = [f for f in features2 if abs(np.radians(f[2]) - a2) < angle_tol]
        
        if not candidates1 or not candidates2:
            continue
        
        # Loop over candidate pairs from features2.
        for f2_1 in candidates1:
            for f2_2 in candidates2:
                if f2_1 == f2_2:
                    continue  # Skip if the same candidate is used twice.
                q1, q2 = f2_1[1], f2_2[1]
                    
                # Avoid degenerate cases.
                if np.linalg.norm(p2 - p1) < 1e-3 or np.linalg.norm(q2 - q1) < 1e-3:
                    continue
                
                # Estimate the candidate transformation using the two point pairs.
                # Assume estimate_transformation_from_2pairs returns a Rigid_Transformation object.
                candidate_transformation = estimate_transformation_from_2pairs(p1, p2, q1, q2)
                
                # Evaluate inliers: transform each feature in features1 and look for a match in features2.
                inliers = []
                for f in features1:
                    idx, pt, ang = f
                    pt_trans = candidate_transformation.rotation_matrix() @ pt + candidate_transformation.translation_vector()
                    matched = False
                    for g in features2:
                        idx2, pt2, ang2 = g
                        if (np.linalg.norm(pt_trans - pt2) < distance_tol and 
                            abs(np.radians(ang) - np.radians(ang2)) < angle_tol):
                            matched = True
                            break
                    if matched:
                        inliers.append(f)
                
                # Update the best transformation if more inliers are found.
                if len(inliers) > best_inlier_count:
                    best_inlier_count = len(inliers)
                    best_transformation = candidate_transformation
                    best_inliers = inliers
                    
    return best_transformation, best_inliers


def main():

    ifc_path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"
    citygml_path = "./test_data/citygml/DEBY_LOD2_4959457.gml"

    # Define two example polygons.

    polygon1 = create_concave_hull(create_IFC_footprint(ifc_path), 0.1) 
    polygon2 = create_CityGML_footprint(citygml_path)

    # polygon1 = np.genfromtxt("./test_data/points/floorplan-vertices-automatic-sparse.txt", dtype=np.double, delimiter=",", encoding="utf-8-sig")
    # polygon2 = np.genfromtxt("./test_data/points/footprint-vertices-automatic-sparse.txt", dtype=np.double, delimiter=",", encoding="utf-8-sig")


    # Detect features (corners) using the turning function.
    features1 = detect_features(polygon1, angle_threshold_deg=30)
    features2 = detect_features(polygon2, angle_threshold_deg=30)

    print(f"Polygon1: {len(features1)} features")
    print(f"Polygon2: {len(features2)} features")
    
    # print("Polygon1 features (index, point, angle):")
    # for f in features1:
    #     print(f)
    # print("\nPolygon2 features (index, point, angle):")
    # for f in features2:
    #     print(f)
    
    # Use RANSAC to find the best rigid transformation aligning polygon1's features to polygon2's features.
    rigid_transformation, inliers = ransac_rigid_transform(features1, features2,
                                             distance_tol=5,
                                             angle_tol_deg=15)
    
    if rigid_transformation is None:
        print("RANSAC failed to find a valid transformation.")
        return
    
    print("\nBest transformation found:")
    print(f"Rotation angle theta [radians]:\n{rigid_transformation.theta}")
    print(f"Translation vector t [x, y]:\n{rigid_transformation.t}")
    print("Number of inliers:", len(inliers))
    
    # Apply the transformation to polygon1.
    polygon1_transformed = rigid_transformation.apply_transformation(polygon1)
    
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
