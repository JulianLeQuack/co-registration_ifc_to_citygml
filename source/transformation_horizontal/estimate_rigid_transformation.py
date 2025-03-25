import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import KDTree
from scipy.optimize import least_squares


def compute_transformation(source_pair, target_pair):
    """
    Compute the 2D rigid transformation (rotation and translation) that aligns 
    source_pair (2x2 array) to target_pair (2x2 array) using the fact that the 
    rotation angle can be computed from the direction of the vectors.
    """
    source_vector = source_pair[1] - source_pair[0]
    target_vector = target_pair[1] - target_pair[0]
    
    # Avoid division by zero if the points are too close
    if np.linalg.norm(source_vector) < 1e-6 or np.linalg.norm(target_vector) < 1e-6:
        return None
    
    # Calculate angles for each vector
    source_angle = np.arctan2(source_vector[1], source_vector[0])
    target_angle = np.arctan2(target_vector[1], target_vector[0])
    theta = target_angle - source_angle

    # Compute rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    # Compute translation so that the first source point aligns with the first target point
    translation = target_pair[0] - R @ source_pair[0]
    return theta, translation


def apply_transformation(params, points):
    """
    Applies a 2D rigid transformation given by params=(theta, t) to a set of points.
    """
    theta, translation = params[0], params[1]
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return (R @ points.T).T + translation


def ransac_registration(source_points, target_points, iterations=1000, threshold=0.2):
    """
    RANSAC-based registration.
    
    - Randomly sample 2 points from the source and 2 from the target.
    - Compute the candidate transformation.
    - Transform the source and count inliers (points that are within the threshold of a target point).
    - Return the best transformation parameters and inlier count.
    """
    best_inliers = 0
    best_params = None
    tree = KDTree(target_points)
    n_source = len(source_points)
    n_target = len(target_points)

    for _ in range(iterations):
        # Randomly sample 2 distinct indices from source_points
        source_indices = np.random.choice(n_source, 3, replace=False)
        # Randomly sample 2 distinct indices from target_points
        target_indices = np.random.choice(n_target, 3, replace=False)
        
        source_pair = source_points[source_indices]
        target_pair = target_points[target_indices]
        
        candidate = compute_transformation(source_pair, target_pair)
        if candidate is None:
            continue
        theta, translation = candidate
        transformed = apply_transformation((theta, translation), source_points)
        
        # For each transformed source point, find the distance to the nearest target point
        distances, _ = tree.query(transformed)
        inliers = np.sum(distances < threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_params = (theta, translation)
    
    return best_params, best_inliers


def icp_residuals(params, source_points, target_points, nn_indices):
    # Transform source points using params to find residuals between transformed source and target
    theta = params[0]
    translation = params[1:3]
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    transformed_source = (R @ source_points.T).T + translation
    # Compute residuals between transformed source points and their current nearest neighbors
    return (transformed_source - target_points[nn_indices]).ravel()


def refine_registration(source_points, target_points, initial_params, max_iterations=100, distance_threshold=10):
    # Extract params from imput params
    params = np.array([initial_params[0], initial_params[1][0], initial_params[1][1]])
    target_tree = KDTree(target_points)

    for _ in range(max_iterations):
        transformed_source = apply_transformation((params[0], params[1:3]), source_points)
        distances, nn_indices = target_tree.query(transformed_source)
        
        valid_mask = distances < distance_threshold
        if np.sum(valid_mask) == 0:
            break
        
        # Optimize using only valid correspondences.
        res = least_squares(
            icp_residuals,
            params,
            args=(
                source_points[valid_mask],
                target_points[nn_indices][valid_mask],
                np.arange(np.sum(valid_mask))
            ),
            loss='huber'
        )
        params = res.x
    
    return params


if __name__ == "__main__":
    source_path = "./test_data/points/floorplan-vertices-automatic-simple-CONVEX.txt"
    target_path = "./test_data/points/footprint-vertices-automatic-simple-CONVEX.txt"
    source = np.genfromtxt(source_path, dtype=np.double, delimiter=",", encoding="utf-8-sig")
    target = np.genfromtxt(target_path, dtype=np.double, delimiter=",", encoding="utf-8-sig")

    rough_transformation, inliers = ransac_registration(source, target, iterations=50000, threshold=0.03) #1
    print(rough_transformation[0], rough_transformation[1][0], rough_transformation[1][1])
    refined_transformation = refine_registration(source, target, rough_transformation, max_iterations=100, distance_threshold=10)
    print(refined_transformation[0], refined_transformation[1:3])

    roughly_transformed_source = apply_transformation(rough_transformation, source)
    refined_transformed_source = apply_transformation(refined_transformation, source)

    plt.figure()
    plt.scatter(source[:, 0], source[:, 1], color='green', label='Source Points', s=10)
    plt.scatter(target[:, 0], target[:, 1], label="Target Points", color="blue", s=10)
    plt.scatter(roughly_transformed_source[:, 0], roughly_transformed_source[:, 1], color="orange", marker="x", s=10)
    plt.scatter(refined_transformed_source[:, 0], refined_transformed_source[:, 1], color="red", marker="x", s=10)
    plt.grid(True)
    plt.show()