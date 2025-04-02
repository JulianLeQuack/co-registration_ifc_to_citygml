import matplotlib.pyplot as plt
import numpy as np

# Import KDTree for efficient nearest neighbor searches
from scipy.spatial import KDTree 
# Import least_squares for optimization-based refinement (like ICP)
from scipy.optimize import least_squares 

# Assuming Rigid_Transformation is defined in './rigid_transformation.py'
# This class likely handles storing (x, y, theta) and applying the transformation.
from source.transformation_horizontal.rigid_transformation import Rigid_Transformation

from source.transformation_horizontal.check_centroid_distance import check_centroid_distance


def compute_transformation(source_pair, target_pair):
    """
    Compute the 2D rigid transformation (rotation and translation) that aligns 
    a specific pair of source points (source_pair) to a corresponding pair 
    of target points (target_pair).

    The method works by aligning the vectors defined by these point pairs.
    The rotation is found by the difference in the angles of these vectors.
    The translation is then calculated to map the first source point to the 
    first target point after rotation.

    Args:
        source_pair (np.ndarray): A 2x2 numpy array [ [x1, y1], [x2, y2] ] 
                                  representing the two source points.
        target_pair (np.ndarray): A 2x2 numpy array [ [x1, y1], [x2, y2] ]
                                  representing the two corresponding target points.

    Returns:
        Rigid_Transformation: An object containing the computed transformation 
                              (translation x, y, rotation theta), or None if 
                              the points in a pair are too close (degenerate case).
    """

    # Calculate the vector between the points in the source pair
    source_vector = source_pair[1] - source_pair[0]
    # Calculate the vector between the points in the target pair
    target_vector = target_pair[1] - target_pair[0]
    
    # Check for degenerate cases: if points in either pair are nearly coincident.
    # This avoids division by zero or numerical instability in angle calculation.
    if np.linalg.norm(source_vector) < 1e-6 or np.linalg.norm(target_vector) < 1e-6:
        # Cannot reliably compute angle/transformation from near-zero length vectors.
        return None 
    
    # Calculate the angle of the source vector relative to the positive x-axis
    # np.arctan2 handles all quadrants correctly.
    source_angle = np.arctan2(source_vector[1], source_vector[0])
    # Calculate the angle of the target vector relative to the positive x-axis
    target_angle = np.arctan2(target_vector[1], target_vector[0])
    
    # The required rotation angle (theta) is the difference between the target and source angles.
    # This rotation will align the source_vector direction with the target_vector direction.
    theta = target_angle - source_angle

    # Compute the 2D rotation matrix R for the calculated angle theta.
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    
    # Compute the translation vector.
    # After rotating the first source point (source_pair[0]) by R, we need to translate it
    # so that it exactly matches the first target point (target_pair[0]).
    # translation = target_point - Rotated_source_point
    translation = target_pair[0] - R @ source_pair[0] 
    
    # Create and return the transformation object using the calculated parameters.
    rigid_transformation = Rigid_Transformation(translation[0], translation[1], theta)
    return rigid_transformation


def ransac_registration(source_points, target_points, iterations=1000, threshold=0.2):
    """
    Performs robust 2D rigid registration using the RANSAC (Random Sample Consensus) algorithm.

    RANSAC works by:
    1. Randomly selecting a minimal set of points needed to estimate the transformation model 
       (2 points for 2D rigid transformation in this implementation, although 3 are sampled).
    2. Computing a candidate transformation from these sampled points.
    3. Transforming all source points using the candidate transformation.
    4. Counting 'inliers': source points whose transformed position is close 
       (within 'threshold') to any target point.
    5. Repeating this process for a fixed number of 'iterations'.
    6. Keeping the transformation that resulted in the highest number of inliers.

    This makes the method robust to outliers and noise, as transformations based on 
    incorrect correspondences are unlikely to produce a high inlier count.

    Args:
        source_points (np.ndarray): Nx2 array of source points.
        target_points (np.ndarray): Mx2 array of target points.
        iterations (int): The number of RANSAC iterations to perform. More iterations 
                          increase the chance of finding a good model but take longer.
        threshold (float): The maximum distance between a transformed source point 
                           and its nearest target point to be considered an inlier.

    Returns:
        tuple: A tuple containing:
            - Rigid_Transformation: The best transformation found.
            - int: The number of inliers corresponding to the best transformation.
    """
    # Initialize the best inlier count found so far to zero.
    best_inliers = 0
    # Initialize the best transformation with default parameters (e.g., identity).
    best_transformation = Rigid_Transformation() 
    # Build a KDTree from the target points for efficient nearest neighbor searches.
    # This significantly speeds up the process of finding distances to target points.
    tree = KDTree(target_points) 
    # Get the number of source and target points.
    n_source = len(source_points)
    n_target = len(target_points)

    # Perform the RANSAC iterations.
    for _ in range(iterations):
        # Randomly sample indices for 2 distinct points from the source points.
        source_indices = np.random.choice(n_source, 2, replace=False) 
        # Randomly sample indices for 2 distinct points from the target points.
        target_indices = np.random.choice(n_target, 2, replace=False)
        
        # Select the actual point pairs based on the sampled indices.
        source_pair = source_points[source_indices] 
        target_pair = target_points[target_indices]
        
        # Compute a candidate transformation using the *first two* sampled points from each set.
        candidate_transformation = compute_transformation(source_pair, target_pair) 
        
        # If compute_transformation failed (e.g., degenerate points), skip this iteration.
        if candidate_transformation is None:
            continue

        # print(candidate_transformation)

        if check_centroid_distance(source_points, target_points, candidate_transformation):
                
            # Apply the candidate transformation to *all* source points.
            transformed = candidate_transformation.apply_transformation(points=source_points)
            
            # Find the distance from each transformed source point to its nearest neighbor 
            # in the target point set using the pre-built KDTree.
            # tree.query returns distances and indices; we only need distances here.
            distances, _ = tree.query(transformed) 
            
            # Count the number of inliers: points where the nearest neighbor distance is below the threshold.
            inliers = np.sum(distances < threshold) 
            
            # If the current candidate transformation produced more inliers than the best one found so far:
            if inliers > best_inliers:
                # Update the best inlier count.
                best_inliers = inliers
                # Update the best transformation.
                best_transformation = candidate_transformation

        else:
            continue
    
    # Return the best transformation found after all iterations and its inlier count.
    return best_transformation, best_inliers


def icp_residuals(params, source_points_subset, target_points_corresponding, nn_indices_subset):
    """
    Calculates the residuals (errors) for the ICP refinement step, designed to be 
    used with `scipy.optimize.least_squares`.

    It computes the vector differences between source points (after being transformed 
    by the current transformation estimate 'params') and their corresponding target points.

    Args:
        params (np.ndarray): A 1D array [tx, ty, theta] representing the current 
                             estimate of the rigid transformation parameters.
        source_points_subset (np.ndarray): The subset of source points (inliers) 
                                           being considered in this optimization step.
        target_points_corresponding (np.ndarray): The target points that correspond 
                                                  to the `source_points_subset`. 
                                                  (i.e., target_points[nn_indices][valid_mask] 
                                                  from `refine_registration`).
        nn_indices_subset (np.ndarray): Indices mapping points in source_points_subset 
                                        to their partners in target_points_corresponding. 
                                        Since target_points_corresponding is already the matched set,
                                        this is typically just np.arange(len(source_points_subset)).


    Returns:
        np.ndarray: A flattened 1D array of residuals (errors) 
                   [e_x1, e_y1, e_x2, e_y2, ...], suitable for `least_squares`.
    """
    # Create a Rigid_Transformation object from the current parameter estimates.
    rigid_transformation = Rigid_Transformation(params[0], params[1], params[2]) 
    
    # Transform the subset of source points using the current transformation estimate.
    transformed_source = rigid_transformation.apply_transformation(source_points_subset)
    
    # Compute the vector differences (residuals) between each transformed source point
    # and its corresponding target point.
    # Note: target_points_corresponding[nn_indices_subset] should yield the correctly ordered target points.
    residuals = transformed_source - target_points_corresponding#[nn_indices_subset] # Indexing might be redundant if target_points_corresponding is already correctly ordered/subsetted
    
    # Flatten the array of residual vectors into a single 1D array [e_x1, e_y1, e_x2, e_y2, ...].
    # This is the required format for scipy.optimize.least_squares.
    return residuals.ravel() 


def refine_registration(source_points, target_points, initial_transformation: Rigid_Transformation, max_iterations=100, distance_threshold=10):
    """
    Refines an initial rigid transformation estimate using an ICP (Iterative Closest Point) 
    -like approach with non-linear least squares optimization.

    It iteratively performs these steps:
    1. Transform source points using the current transformation estimate.
    2. Find nearest neighbors in the target set for the transformed source points.
    3. Filter correspondences: Keep only pairs where the distance is below `distance_threshold`.
    4. Optimize the transformation parameters (tx, ty, theta) using `least_squares` 
       to minimize the distances between these valid corresponding points (using `icp_residuals`).
    5. Update the transformation estimate.
    6. Repeat until convergence (implicitly handled by least_squares over iterations) or max_iterations.

    Args:
        source_points (np.ndarray): Nx2 array of all source points.
        target_points (np.ndarray): Mx2 array of all target points.
        initial_transformation (Rigid_Transformation): An initial estimate of the 
                                                     transformation (e.g., from RANSAC).
        max_iterations (int): Maximum number of refinement iterations.
        distance_threshold (float): Maximum distance for a point pair to be considered 
                                   a valid correspondence during the optimization phase. 
                                   Helps reject outliers or points from non-overlapping parts.

    Returns:
        Rigid_Transformation: The refined transformation object.
    """
    # Start with the provided initial transformation.
    current_transformation = initial_transformation
    # Extract the numerical parameters [tx, ty, theta] for the optimizer.
    params = np.array([initial_transformation.x, initial_transformation.y, initial_transformation.theta])
    # Build a KDTree for the target points for efficient nearest neighbor searches within the loop.
    target_tree = KDTree(target_points)

    # Perform the refinement iterations.
    for i in range(max_iterations):
        # Transform *all* source points using the *current* transformation estimate.
        transformed_source = current_transformation.apply_transformation(points=source_points)
        
        # Find the nearest target point and distance for each transformed source point.
        distances, nn_indices = target_tree.query(transformed_source)
        
        # Create a boolean mask to select only 'valid' correspondences, 
        # i.e., pairs where the distance is below the specified threshold.
        # This step is crucial for robustness, excluding pairs that are likely outliers
        # or belong to non-overlapping parts of the footprints (like appendices).
        valid_mask = distances < distance_threshold
        
        # Count the number of valid correspondences found in this iteration.
        num_valid = np.sum(valid_mask)
        
        # If no valid correspondences are found (e.g., shapes moved too far apart, 
        # threshold too strict), stop the refinement process.
        if num_valid == 0:
            print(f"Refinement stopped at iteration {i+1}: No valid correspondences found below threshold {distance_threshold}.")
            break 
        
        # Select the subset of source points that have valid correspondences.
        source_points_subset = source_points[valid_mask]
        # Select the corresponding target points using the nearest neighbor indices.
        target_points_corresponding = target_points[nn_indices[valid_mask]]
        
        # Perform the non-linear least squares optimization.
        # It tries to find the 'params' (tx, ty, theta) that minimize the sum of squares 
        # of the residuals calculated by 'icp_residuals'.
        res = least_squares(
            fun=icp_residuals, # The function calculating residuals.
            x0=params,          # The initial guess for parameters (from previous iteration).
            args=(              # Additional arguments passed to icp_residuals:
                source_points_subset,        # The source points to transform.
                target_points_corresponding, # Their corresponding target points.
                np.arange(num_valid)         # Indices mapping subset source to subset target (0 to N-1).
            ),
            loss='huber',     # Use a robust loss function ('huber') which is less sensitive 
                              # to remaining outliers than standard linear ('ls') loss.
            jac='3-point'     # Method to estimate Jacobian; '3-point' is often a good balance.
                              # Could also use '2-point' or provide an analytical Jacobian if known.
        )
        
        # Update the transformation parameters with the result from the optimization.
        params = res.x 
        # Update the transformation object with the new parameters.
        current_transformation = Rigid_Transformation(params[0], params[1], params[2])
    
    # Return the final refined transformation after all iterations or early stopping.
    return current_transformation


# This block executes only when the script is run directly (not imported as a module).
if __name__ == "__main__":

    # Import functions assumed to be defined elsewhere for creating footprints 
    # from various file formats or hull algorithms.
    from ...source.transformation_horizontal.create_footprints.create_hull import create_concave_hull, create_convex_hull
    from ...source.transformation_horizontal.create_footprints.create_CityGML_footprint import create_CityGML_footprint
    from ...source.transformation_horizontal.create_footprints.create_DXF_footprint import create_DXF_footprint
    from ...source.transformation_horizontal.create_footprints.create_IFC_footprint import create_IFC_footprint

    # --- Example Data Loading ---
    # Define file paths for input data (IFC, DXF, CityGML).
    ifc_path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"
    floorplan_path = "./test_data/dxf/01-05-0501_EG.dxf"
    citygml_path = "./test_data/citygml/DEBY_LOD2_4959457.gml"

    # Create the source point set: 
    # 1. Extract footprint vertices/points from an IFC file using create_IFC_footprint.
    # 2. Compute a concave hull (alpha shape) of these points with alpha=0.1 using create_concave_hull.
    #    This likely generates boundary points for the footprint.
    source = create_concave_hull(create_IFC_footprint(ifc_path), 0.1) 
    # Create the target point set:
    # Extract footprint vertices/points from a CityGML file using create_CityGML_footprint.
    target = create_CityGML_footprint(citygml_path)

    # --- Registration Pipeline ---
    # 1. Perform RANSAC to get a robust but potentially rough initial alignment.
    #    Using a high number of iterations (50000) and a tight threshold (0.03) 
    #    suggests the initial alignment might be significantly off or noisy data is expected.
    print("Running RANSAC for rough registration...")
    rough_transformation, inliers = ransac_registration(source, target, iterations=25000, threshold=0.03) 
    print("RANSAC complete. Best transformation found:")
    print(rough_transformation) # Print the rough transformation parameters.
    print(f"Number of inliers for rough transformation: {inliers}")

    # 2. Refine the rough transformation using the ICP-like optimization.
    #    Use the output of RANSAC as the starting point.
    #    The distance_threshold (5) here is much larger than the RANSAC threshold,
    #    allowing more point pairs (within 5 units) to influence the refinement.
    print("\nRunning refinement step...")
    refined_transformation = refine_registration(source, target, rough_transformation, max_iterations=100, distance_threshold=5)
    print("Refinement complete. Final transformation:")
    print(refined_transformation) # Print the refined transformation parameters.

    # --- Apply Transformations for Visualization ---
    # Apply the rough RANSAC transformation to the original source points.
    roughly_transformed_source = rough_transformation.apply_transformation(points=source) 
    # Apply the final refined transformation to the original source points.
    refined_transformed_source = refined_transformation.apply_transformation(points=source)

    # --- Plotting Results ---
    print("\nGenerating plot...")
    plt.figure(figsize=(18, 6)) # Create a figure with 3 subplots.

    # Subplot 1: Original Source Points
    plt.subplot(1, 3, 1)
    plt.scatter(source[:, 0], source[:, 1], color='green', label='Source Points (IFC Hull)', s=10)
    plt.title("Source Input (IFC Concave Hull)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.axis('equal') # Ensure aspect ratio is equal for correct shape visualization.
    plt.legend()

    # Subplot 2: Original Target Points
    plt.subplot(1, 3, 2)
    plt.scatter(target[:, 0], target[:, 1], label="Target Points (CityGML)", color="blue", s=10)
    plt.title("Target Input (CityGML Footprint)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()

    # Subplot 3: Target, Roughly Transformed Source, and Refined Transformed Source
    plt.subplot(1, 3, 3)
    plt.scatter(target[:, 0], target[:, 1], label="Target Points", color="blue", s=10, alpha=0.6) # Target slightly transparent
    plt.scatter(roughly_transformed_source[:, 0], roughly_transformed_source[:, 1], color="orange", marker=".", s=10, label="Rough (RANSAC) Registration") # Use dots for rough
    plt.scatter(refined_transformed_source[:, 0], refined_transformed_source[:, 1], color="red", marker="x", s=15, label="Refined Registration") # Use crosses for refined
    plt.title("Registration Result")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Adjust layout to prevent labels/titles overlapping.
    plt.tight_layout() 
    # Display the plot.
    plt.show()
    print("Plot displayed.")