import numpy as np
import matplotlib.pyplot as plt

def detect_corners(points, threshold_angle_deg=30):
    """
    Detects corner features in a set of points representing a polygon.
    
    For each point, the function computes the turning angle between the vector from the previous point
    to the current point and the vector from the current point to the next point.
    
    Parameters:
      points (np.ndarray): An array of shape (N,2) representing the polygon vertices.
      threshold_angle_deg (float): Only points with an absolute turning angle greater than this 
                                   threshold (in degrees) are marked as corners.
    
    Returns:
      corners (list): Indices of the points identified as corners.
      angles (list): The corresponding turning angles (in degrees) at those points.
    """
    corners = []
    angles = []
    n = len(points)
    for i in range(n):
        # Use modulo arithmetic for closed polygons (last point connects to first)
        prev = points[i - 1]
        curr = points[i]
        next = points[(i + 1) % n]
        
        # Compute the vectors from the previous to current and from current to next
        v1 = curr - prev
        v2 = next - curr
        
        # Calculate the signed turning angle using arctan2.
        # The determinant gives the sine component while the dot product gives the cosine.
        angle_rad = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
        angle_deg = np.degrees(angle_rad)
        
        # If the turning angle exceeds the threshold, mark this vertex as a corner.
        if np.abs(angle_deg) > threshold_angle_deg:
            corners.append(i)
            angles.append(angle_deg)
    
    return corners, angles

def rigid_transform(A, B):
    """
    Computes the best-fit rigid transformation (rotation and translation)
    that aligns point set A to point set B using the Kabsch algorithm.
    
    Parameters:
      A (np.ndarray): Source points, shape (N,2).
      B (np.ndarray): Destination points, shape (N,2).
    
    Returns:
      R (np.ndarray): 2x2 rotation matrix.
      t (np.ndarray): Translation vector of shape (2,).
    """
    # Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # Center the points
    AA = A - centroid_A
    BB = B - centroid_B
    
    # Compute the covariance matrix
    H = AA.T @ BB
    
    # Compute the Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(H)
    
    # Compute the rotation matrix
    R = Vt.T @ U.T
    
    # Ensure a proper rotation (determinant = +1). If not, fix it.
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute the translation vector
    t = centroid_B - R @ centroid_A
    
    return R, t

def apply_transform(points, R, t):
    """
    Applies the rigid transformation defined by rotation R and translation t to the set of points.
    
    Parameters:
      points (np.ndarray): Array of shape (N,2).
      R (np.ndarray): 2x2 rotation matrix.
      t (np.ndarray): Translation vector.
    
    Returns:
      transformed (np.ndarray): The transformed set of points.
    """
    return (R @ points.T).T + t

def main():
    # # Example input: Two polygons represented by a set of points.
    # # Note: These polygons are similar but not identical. They share some common local features.
    # polygon1 = np.array([
    #     [0, 0],
    #     [1, 0],
    #     [1.5, 1],
    #     [1, 2],
    #     [0, 2],
    #     [-0.5, 1]
    # ])
    
    # polygon2 = np.array([
    #     [2, 1],
    #     [3, 1],
    #     [3.5, 2],
    #     [3, 3],
    #     [2, 3],
    #     [1.5, 2]
    # ])

    polygon1 = np.genfromtxt("./test_data/points/floorplan-vertices-automatic-sparse.txt", dtype=np.double, delimiter=",", encoding="utf-8-sig")
    polygon2 = np.genfromtxt("./test_data/points/footprint-vertices-automatic-sparse.txt", dtype=np.double, delimiter=",", encoding="utf-8-sig")


    # --- Step 1: Local Feature Detection ---
    # Detect corners based on a turning angle threshold.
    corners1, angles1 = detect_corners(polygon1, threshold_angle_deg=30)
    corners2, angles2 = detect_corners(polygon2, threshold_angle_deg=30)
    
    print("Polygon1 corners indices:", corners1, "with turning angles (deg):", angles1)
    print("Polygon2 corners indices:", corners2, "with turning angles (deg):", angles2)
    
    # --- Step 2: Feature Correspondence ---
    # For simplicity, assume that the detected corners are in the same order in both polygons.
    # In practice, more sophisticated feature matching (e.g., using additional descriptors) might be needed.
    feature_points1 = polygon1[corners1]
    feature_points2 = polygon2[corners2]
    
    # --- Step 3: Compute Rigid Transformation ---
    # Use the Kabsch algorithm to compute the rotation and translation that best aligns the feature points.
    R, t = rigid_transform(feature_points1, feature_points2)
    print("Computed Rotation Matrix R:\n", R)
    print("Computed Translation Vector t:\n", t)
    
    # --- Step 4: Apply the Transformation ---
    # Align the entire polygon1 to polygon2 using the computed rigid transformation.
    polygon1_aligned = apply_transform(polygon1, R, t)
    
    # --- Step 5: Visualization ---
    # Plot the original polygon1, the aligned polygon1, and polygon2 to visualize the alignment.
    plt.figure(figsize=(8, 6))
    plt.plot(polygon1[:, 0], polygon1[:, 1], 'bo-', label='Polygon1 (Original)')
    plt.plot(polygon1_aligned[:, 0], polygon1_aligned[:, 1], 'ro-', label='Polygon1 (Aligned)')
    plt.plot(polygon2[:, 0], polygon2[:, 1], 'go-', label='Polygon2')
    
    # Mark the detected corner features for clarity.
    plt.plot(feature_points1[:, 0], feature_points1[:, 1], 'kx', markersize=10, label='Polygon1 Corners')
    plt.plot(feature_points2[:, 0], feature_points2[:, 1], 'k+', markersize=10, label='Polygon2 Corners')
    
    plt.title("Rigid Alignment of Two Polygons")
    plt.legend()
    plt.axis('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

if __name__ == "__main__":
    main()
