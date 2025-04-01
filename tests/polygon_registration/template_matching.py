import numpy as np
import matplotlib.pyplot as plt

def simplify_polygon_turning(polygon, threshold=0.0873):
    """
    Simplify a polygon using a turning function approach.
    For each vertex, compute the turning (angle change) from the previous to the next vertex.
    Vertices with a small turning angle (below the threshold, in radians) are considered insignificant.
    
    Parameters:
        polygon: (N, 2) numpy array of vertices (assumed to be ordered).
        threshold: Minimum turning angle (in radians) to retain a vertex (default ~5 degrees).
    
    Returns:
        simplified: A (M, 2) numpy array of the simplified polygon vertices.
    """
    # Ensure polygon is closed: if the first and last points are not the same, close the polygon.
    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])
    
    N = len(polygon) - 1  # last point is same as first
    keep_indices = [0]  # always keep the first vertex
    
    for i in range(1, N):
        prev = polygon[i - 1]
        curr = polygon[i]
        next_pt = polygon[(i + 1) % N]
        
        v1 = curr - prev
        v2 = next_pt - curr
        
        # Compute the signed turning angle at the current vertex using arctan2.
        angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
        if np.abs(angle) >= threshold:
            keep_indices.append(i)
    
    # Ensure at least three points are kept for a valid polygon.
    if len(keep_indices) < 3:
        keep_indices = list(range(N))
    
    simplified = polygon[keep_indices]
    # Ensure closure by appending the first vertex if needed.
    if not np.allclose(simplified[0], simplified[-1]):
        simplified = np.vstack([simplified, simplified[0]])
    
    return simplified

def resample_polygon(polygon, num_points=50):
    """
    Resample a polygon uniformly along its perimeter.
    
    Parameters:
        polygon: (N, 2) numpy array of vertices (assumed closed).
        num_points: Number of points for resampling.
    
    Returns:
        resampled: (num_points, 2) numpy array of uniformly sampled points.
    """
    # Compute distances between consecutive vertices.
    diffs = np.diff(polygon, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative = np.hstack([[0], np.cumsum(dists)])
    total_length = cumulative[-1]
    
    # Evenly spaced sample distances.
    sample_dists = np.linspace(0, total_length, num_points)
    resampled = []
    
    for sd in sample_dists:
        # Find the segment where sd falls.
        idx = np.searchsorted(cumulative, sd) - 1
        idx = np.clip(idx, 0, len(diffs)-1)
        t = (sd - cumulative[idx]) / dists[idx]
        pt = polygon[idx] * (1 - t) + polygon[idx+1] * t
        resampled.append(pt)
        
    return np.array(resampled)

def rigid_transform_2D(source, target):
    """
    Estimate the rigid transformation (rotation and translation only) that maps
    the source points to the target points using the Kabsch algorithm.
    
    Parameters:
        source: (N, 2) numpy array of source points.
        target: (N, 2) numpy array of corresponding target points.
    
    Returns:
        R: 2x2 rotation matrix.
        t: Translation vector (length 2).
    """
    # Compute centroids.
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)
    
    # Center the points.
    source_centered = source - centroid_source
    target_centered = target - centroid_target
    
    # Compute covariance matrix.
    H = source_centered.T.dot(target_centered)
    
    # SVD.
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    
    # Correct for reflection.
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T.dot(U.T)
    
    # Compute translation.
    t = centroid_target - R.dot(centroid_source)
    
    return R, t

def apply_transformation(points, R, t):
    """
    Apply the rigid transformation to a set of points.
    
    Parameters:
        points: (N,2) numpy array of (x, y) coordinates.
        R: 2x2 rotation matrix.
        t: Translation vector.
    
    Returns:
        Transformed points as an (N,2) numpy array.
    """
    return (R.dot(points.T)).T + t

def plot_alignment(target, source, aligned, title="Alignment of Building Footprints"):
    """
    Plot the target, original source, and aligned source footprints.
    """
    plt.figure(figsize=(8, 6))
    # Close the polygons for plotting.
    target_plot = np.vstack([target, target[0]])
    source_plot = np.vstack([source, source[0]])
    aligned_plot = np.vstack([aligned, aligned[0]])
    
    plt.plot(target_plot[:, 0], target_plot[:, 1], 'bo-', linewidth=2, label='Target Footprint')
    plt.plot(source_plot[:, 0], source_plot[:, 1], 'ro--', linewidth=2, label='Source Footprint (Original)')
    plt.plot(aligned_plot[:, 0], aligned_plot[:, 1], 'go-', linewidth=2, label='Source Footprint (Aligned)')
    
    plt.legend()
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

# --------------------------
# Example Usage
# --------------------------

source_footprint = np.genfromtxt("./test_data/points/floorplan-vertices-automatic-sparse.txt", dtype=np.double, delimiter=",", encoding="utf-8-sig")
target_footprint = np.genfromtxt("./test_data/points/footprint-vertices-automatic-sparse.txt", dtype=np.double, delimiter=",", encoding="utf-8-sig")

# Simplify the polygons using the turning function.
threshold = 0.0873  # about 5 degrees in radians
target_simplified = simplify_polygon_turning(target_footprint, threshold)
source_simplified = simplify_polygon_turning(source_footprint, threshold)

# Resample the simplified polygons to have the same number of points.
num_sample_points = 50
target_resampled = resample_polygon(target_simplified, num_sample_points)
source_resampled = resample_polygon(source_simplified, num_sample_points)

# Estimate the rigid transformation (rotation and translation) between the resampled footprints.
R_est, t_est = rigid_transform_2D(source_resampled, target_resampled)
print("Estimated rotation matrix:")
print(R_est)
print("Estimated translation vector:")
print(t_est)

# Apply the estimated transformation to the original (or resampled) source polygon.
aligned_resampled = apply_transformation(source_resampled, R_est, t_est)

# Plot the simplified and resampled target, original source, and aligned source footprints.
plot_alignment(target_resampled, source_resampled, aligned_resampled,
               title="Rigid Alignment after Turning-Function Simplification")
