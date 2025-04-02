import numpy as np
import matplotlib.pyplot as plt

def resample_polygon(points, num_points=100):
    """
    Resamples a closed polygon so that it has num_points equally spaced
    along its perimeter. This removes any dependency on the original vertex count.
    
    Parameters:
      points (np.ndarray): Original polygon vertices, shape (N,2).
      num_points (int): Number of points to resample.
      
    Returns:
      new_points (np.ndarray): Resampled polygon vertices, shape (num_points,2).
    """
    # Compute distances between consecutive points (using the polygon closure)
    distances = np.sqrt(np.sum(np.diff(points, axis=0, append=[points[0]])**2, axis=1))
    cumulative = np.cumsum(distances)
    total_length = cumulative[-1]
    # Insert 0 at the beginning for cumulative distance
    cumulative = np.insert(cumulative, 0, 0)
    
    # Create equally spaced distances along the perimeter
    sample_distances = np.linspace(0, total_length, num_points, endpoint=False)
    
    new_points = []
    # For each desired distance, find its location on the original polygon using linear interpolation.
    for d in sample_distances:
        # Find segment where d falls
        idx = np.searchsorted(cumulative, d) - 1
        idx = np.clip(idx, 0, len(points)-1)
        # Compute interpolation factor t in the segment between points[idx] and points[idx+1]
        t = (d - cumulative[idx]) / (cumulative[idx+1] - cumulative[idx])
        p = (1 - t) * points[idx] + t * points[(idx + 1) % len(points)]
        new_points.append(p)
    return np.array(new_points)

def compute_turning_function(points):
    """
    Computes the turning function for a closed polygon.
    
    The turning function is defined as the cumulative sum of the turning angles
    at each vertex. For each vertex, the turning angle is computed as the angle 
    between the incoming and outgoing edge directions.
    
    Parameters:
      points (np.ndarray): Polygon vertices, assumed to be in order, shape (N,2).
      
    Returns:
      turning_function (np.ndarray): Cumulative turning angle at each point.
    """
    N = len(points)
    angles = []
    for i in range(N):
        prev = points[i - 1]
        curr = points[i]
        nxt = points[(i + 1) % N]
        # Compute vectors for the incoming and outgoing edges
        v1 = curr - prev
        v2 = nxt - curr
        # Compute the signed turning angle using arctan2 (using the determinant and dot product)
        angle = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
        angles.append(angle)
    angles = np.array(angles)
    # The turning function is the cumulative sum of the angles.
    turning_function = np.cumsum(angles)
    # Normalize so that it starts at 0
    turning_function -= turning_function[0]
    return turning_function

def best_cyclic_shift(turning1, turning2):
    """
    Finds the best cyclic (circular) shift that aligns two turning functions.
    
    The function tests every possible circular shift of turning1 and computes the sum of squared
    differences with turning2. The shift with the minimal error is chosen.
    
    Parameters:
      turning1 (np.ndarray): First turning function.
      turning2 (np.ndarray): Second turning function.
      
    Returns:
      best_shift (int): Number of indices to roll turning1 so that it best aligns with turning2.
    """
    N = len(turning1)
    best_shift = 0
    best_error = np.inf
    for shift in range(N):
        shifted = np.roll(turning1, shift)
        error = np.sum((shifted - turning2)**2)
        if error < best_error:
            best_error = error
            best_shift = shift
    return best_shift

def rigid_transform(A, B):
    """
    Computes the best-fit rigid transformation (rotation and translation)
    that aligns point set A to point set B using the Kabsch algorithm.
    
    Note: A and B must have the same number of points.
    
    Parameters:
      A (np.ndarray): Source points, shape (N,2).
      B (np.ndarray): Destination points, shape (N,2).
      
    Returns:
      R (np.ndarray): 2x2 rotation matrix.
      t (np.ndarray): Translation vector (2,).
    """
    # Compute centroids of each point set
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # Center the points
    AA = A - centroid_A
    BB = B - centroid_B
    
    # Compute the covariance matrix
    H = AA.T @ BB
    # Singular Value Decomposition
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure a proper rotation (determinant should be +1)
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    
    t = centroid_B - R @ centroid_A
    return R, t

def apply_transform(points, R, t):
    """
    Applies the rigid transformation (rotation R and translation t) to a set of points.
    
    Parameters:
      points (np.ndarray): Points to transform, shape (N,2).
      R (np.ndarray): 2x2 rotation matrix.
      t (np.ndarray): Translation vector.
      
    Returns:
      transformed_points (np.ndarray): Transformed points.
    """
    return (R @ points.T).T + t

def main():
    # Example: two polygons with different numbers of vertices.
    # # (These can be replaced with your own input data.)
    
    # # Define polygon1 (e.g., a hexagon-like shape)
    # polygon1 = np.array([
    #     [0, 0],
    #     [1, 0],
    #     [1.5, 1],
    #     [1, 2],
    #     [0, 2],
    #     [-0.5, 1]
    # ])
    
    # # Define polygon2 (a similar shape but with a different vertex count and sampling)
    # polygon2 = np.array([
    #     [2, 1],
    #     [2.5, 0.5],
    #     [3, 1],
    #     [3.2, 1.8],
    #     [3, 2.5],
    #     [2, 2.5],
    #     [1.8, 1.8],
    #     [1.5, 1]
    # ])
    

    polygon1 = np.genfromtxt("./test_data/points/floorplan-vertices-automatic-sparse.txt", dtype=np.double, delimiter=",", encoding="utf-8-sig")
    polygon2 = np.genfromtxt("./test_data/points/footprint-vertices-automatic-sparse.txt", dtype=np.double, delimiter=",", encoding="utf-8-sig")

    # Step 1: Resample both polygons to a fixed number of points.
    num_samples = 100
    poly1_resampled = resample_polygon(polygon1, num_points=num_samples)
    poly2_resampled = resample_polygon(polygon2, num_points=num_samples)
    
    # Step 2: Compute the turning functions for each resampled polygon.
    tf1 = compute_turning_function(poly1_resampled)
    tf2 = compute_turning_function(poly2_resampled)
    
    # Step 3: Find the best cyclic shift to align the turning functions.
    shift = best_cyclic_shift(tf1, tf2)
    print("Best cyclic shift (in number of points):", shift)
    
    # Apply the cyclic shift to polygon1 (both the resampled points and turning function)
    poly1_aligned_order = np.roll(poly1_resampled, shift, axis=0)
    tf1_aligned = np.roll(tf1, shift)
    
    # Step 4: Compute the rigid transformation between the aligned polygon1 and polygon2.
    # (Now that both polygons have the same number of points and their local features are aligned,
    #  we can use all points for a robust estimation.)
    R, t = rigid_transform(poly1_aligned_order, poly2_resampled)
    print("Rotation matrix:\n", R)
    print("Translation vector:\n", t)
    
    # Apply the transformation to the original resampled polygon1.
    poly1_transformed = apply_transform(poly1_resampled, R, t)
    
    # Step 5: Visualization
    plt.figure(figsize=(8, 6))
    plt.plot(poly1_resampled[:, 0], poly1_resampled[:, 1], 'bo-', label="Polygon1 Resampled")
    plt.plot(poly2_resampled[:, 0], poly2_resampled[:, 1], 'go-', label="Polygon2 Resampled")
    plt.plot(poly1_transformed[:, 0], poly1_transformed[:, 1], 'ro-', label="Polygon1 Transformed")
    plt.axis('equal')
    plt.title("Rigid Alignment Invariant to Vertex Count")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
