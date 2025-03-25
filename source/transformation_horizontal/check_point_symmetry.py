import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def check_point_symmetry(points: np.array, tolerance=1e-6):

    # Get Centroid of Points as "Mirroring Point"
    centroid = np.mean(points, axis=0)
    reflected_points = 2 * centroid - points

    # Find nearest neighbors to check is projected point lands on point-symmetric other point within a threshhold
    tree = cKDTree(points)
    distances, _ = tree.query(reflected_points, k=1)

    # return True if points create a point-symmetric shape
    return np.all(distances < tolerance)

if __name__ == "__main__":
    point_symmetric = np.array([[-5.066, -3.282],
                               [ 3.366,  1.218],
                               [15.066, 13.282],
                               [ 6.634,  8.782]])
    
    non_point_symmetric = np.array([[ 0.0,  0.0],
                                    [10.0,  2.0],
                                    [ 9.0, 25.0],
                                    [ 1.0, 22.0]])

    plt.figure(figsize=(10,10))
    x_p = point_symmetric[:, 0]
    y_p = point_symmetric[:, 1]
    x_n = non_point_symmetric[:, 0]
    y_n = non_point_symmetric[:, 1]
    plt.scatter(x_p, y_p, color="blue", label=f"Point-symmetric: {check_point_symmetry(point_symmetric)}")
    plt.scatter(x_n, y_n, color="red", label=f"Point-symmetric: {check_point_symmetry(non_point_symmetric)}")
    plt.grid(True)
    plt.legend()
    plt.show()