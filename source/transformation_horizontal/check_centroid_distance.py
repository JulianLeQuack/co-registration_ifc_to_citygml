import numpy as np
import matplotlib.pyplot as plt
from .rigid_transformation import Rigid_Transformation


def calculate_centroid(points: np.array):
    return np.mean(points, axis=0)


def calculate_avg_distance(points: np.array, centroid: np.array):
    distances = np.linalg.norm(points - centroid, axis=1)
    avg_distance = np.mean(distances)
    return avg_distance


def check_centroid_distance(source_points: np.array, target_points: np.array, transformation: Rigid_Transformation):
    '''
    Used to check whether the distance between the transformed source controid and the target centroid is smaller than
    the average distance between the untransformed source centroid and all the source points.
    This is supposed to only run the RANSAC transformation for all source points, if this is the case.
    This rule means that a solution, where the two input polygons are not at all overlapped is not considered.
    '''

    source_centroid = calculate_centroid(source_points)
    target_centroid = calculate_centroid(target_points)
    transformed_source_centroid = transformation.apply_transformation(source_centroid)

    avg_distance = calculate_avg_distance(source_points, source_centroid)
    centroid_distance = np.linalg.norm(transformed_source_centroid - target_centroid)

    # print("Centroid Distance: ", centroid_distance, "\nAvg Distance: ", avg_distance, "\n")

    if centroid_distance <= avg_distance:
        return True
    else:
        return False
    

if __name__ == "__main__":
    # --- Case 1: Returns True ---
    source_points_true = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    target_points_true = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]])
    transformation_true = Rigid_Transformation(0.5, 0.5, 0)

    result_true = check_centroid_distance(source_points_true, target_points_true, transformation_true)
    print(f"Case 1 Result: {result_true}")

    source_centroid_true = calculate_centroid(source_points_true)
    target_centroid_true = calculate_centroid(target_points_true)
    transformed_source_centroid_true = transformation_true.apply_transformation(np.array(source_centroid_true))
    avg_distance_true = calculate_avg_distance(source_points_true, source_centroid_true)
    centroid_distance_true = np.linalg.norm(transformed_source_centroid_true - target_centroid_true)

    # --- Case 2: Returns False ---
    source_points_false = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    target_points_false = np.array([[3, 3], [4, 3], [4, 4], [3, 4]])
    # Apply a transformation that moves the source far from the target
    transformation_false = Rigid_Transformation(7, 7, 0)

    result_false = check_centroid_distance(source_points_false, target_points_false, transformation_false)
    print(f"Case 2 Result: {result_false}")

    source_centroid_false = calculate_centroid(source_points_false)
    target_centroid_false = calculate_centroid(target_points_false)
    transformed_source_centroid_false = transformation_false.apply_transformation(np.array(source_centroid_false))
    avg_distance_false = calculate_avg_distance(source_points_false, source_centroid_false)
    centroid_distance_false = np.linalg.norm(transformed_source_centroid_false - target_centroid_false)

    # --- Plotting ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Case Returns True
    axs[0].scatter(source_points_true[:, 0], source_points_true[:, 1], color='blue', label='Source Points')
    axs[0].scatter(target_points_true[:, 0], target_points_true[:, 1], color='red', label='Target Points')
    axs[0].scatter(source_centroid_true[0], source_centroid_true[1], color='lightblue', marker='o', s=100, label='Source Centroid')
    axs[0].scatter(target_centroid_true[0], target_centroid_true[1], color='salmon', marker='o', s=100, label='Target Centroid')
    axs[0].scatter(transformed_source_centroid_true[0], transformed_source_centroid_true[1], color='green', marker='x', s=100, label='Transformed Source Centroid')
    axs[0].set_title(f'Case: Centroid Distance <= Avg Spread (True)\nCentroid Dist: {centroid_distance_true:.2f}, Avg Spread: {avg_distance_true:.2f}')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_aspect('equal', adjustable='box')

    # Subplot 2: Case Returns False
    axs[1].scatter(source_points_false[:, 0], source_points_false[:, 1], color='blue', label='Source Points')
    axs[1].scatter(target_points_false[:, 0], target_points_false[:, 1], color='red', label='Target Points')
    axs[1].scatter(source_centroid_false[0], source_centroid_false[1], color='lightblue', marker='o', s=100, label='Source Centroid')
    axs[1].scatter(target_centroid_false[0], target_centroid_false[1], color='salmon', marker='o', s=100, label='Target Centroid')
    axs[1].scatter(transformed_source_centroid_false[0], transformed_source_centroid_false[1], color='green', marker='x', s=100, label='Transformed Source Centroid')
    axs[1].set_title(f'Case: Centroid Distance > Avg Spread (False)\nCentroid Dist: {centroid_distance_false:.2f}, Avg Spread: {avg_distance_false:.2f}')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()