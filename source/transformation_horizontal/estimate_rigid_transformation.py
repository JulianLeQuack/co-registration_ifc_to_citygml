import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import KDTree
from scipy.optimize import least_squares

from .rigid_transformation import Rigid_Transformation
from .create_footprints.create_CityGML_footprint import create_CityGML_footprint
from .create_footprints.create_DXF_footprint import create_DXF_footprint
from .create_footprints.create_IFC_footprint import create_IFC_footprint


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
    rigid_transformation = Rigid_Transformation(translation[0], translation[1], theta)
    return rigid_transformation


def ransac_registration(source_points, target_points, iterations=1000, threshold=0.2):
    """
    RANSAC-based registration.
    
    - Randomly sample 2 points from the source and 2 from the target.
    - Compute the candidate transformation.
    - Transform the source and count inliers (points that are within the threshold of a target point).
    - Return the best transformation parameters and inlier count.
    """
    best_inliers = 0
    best_transformation = Rigid_Transformation()
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
        
        candidate_transformation = compute_transformation(source_pair, target_pair)
        if candidate_transformation is None:
            continue
        transformed = candidate_transformation.apply_transformation(points=source_points)
        
        # For each transformed source point, find the distance to the nearest target point
        distances, _ = tree.query(transformed)
        inliers = np.sum(distances < threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_transformation = candidate_transformation
    
    return best_transformation, best_inliers


def icp_residuals(params, source_points, target_points, nn_indices):
    rigid_transformation = Rigid_Transformation(params[0], params[1], params[2])
    # Transform source points using params to find residuals between transformed source and target
    transformed_source = rigid_transformation.apply_transformation(source_points)
    # Compute residuals between transformed source points and their current nearest neighbors
    return (transformed_source - target_points[nn_indices]).ravel()


def refine_registration(source_points, target_points, initial_transformation: Rigid_Transformation, max_iterations=100, distance_threshold=10):
    # Extract params from imput params
    current_transformation = initial_transformation
    params = np.array([initial_transformation.x, initial_transformation.y, initial_transformation.theta])
    target_tree = KDTree(target_points)

    for _ in range(max_iterations):
        transformed_source = current_transformation.apply_transformation(points=source_points)
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
        current_transformation = Rigid_Transformation(params[0], params[1], params[2])
    
    return current_transformation


if __name__ == "__main__":

    from .create_footprints.create_hull import create_concave_hull, create_convex_hull
    # source_path = "./test_data/points/floorplan-vertices-automatic-simple-CONVEX.txt"
    # target_path = "./test_data/points/footprint-vertices-automatic-simple-CONVEX.txt"
    # source = np.genfromtxt(source_path, dtype=np.double, delimiter=",", encoding="utf-8-sig")
    # target = np.genfromtxt(target_path, dtype=np.double, delimiter=",", encoding="utf-8-sig")

    ifc_path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"
    floorplan_path = "./test_data/dxf/01-05-0501_EG.dxf"
    citygml_path = "./test_data/citygml/DEBY_LOD2_4959457.gml"

    source = create_concave_hull(create_IFC_footprint(ifc_path), 0.1)
    target = create_CityGML_footprint(citygml_path)

    rough_transformation, inliers = ransac_registration(source, target, iterations=50000, threshold=0.03)
    print(rough_transformation)
    refined_transformation = refine_registration(source, target, rough_transformation, max_iterations=100, distance_threshold=5)
    print(refined_transformation)

    roughly_transformed_source = rough_transformation.apply_transformation(points=source)
    refined_transformed_source = refined_transformation.apply_transformation(points=source)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.scatter(source[:, 0], source[:, 1], color='green', label='Source Points', s=10)
    plt.title("IFC Input")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.scatter(target[:, 0], target[:, 1], label="Target Points", color="blue", s=10)
    plt.title("CityGML Input")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.scatter(target[:, 0], target[:, 1], label="Target Points", color="blue", s=10)
    plt.scatter(roughly_transformed_source[:, 0], roughly_transformed_source[:, 1], color="orange", marker="x", s=10, label="Rough Registration")
    plt.scatter(refined_transformed_source[:, 0], refined_transformed_source[:, 1], color="red", marker="x", s=10, label="Refined Registration")
    plt.title("Result")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()