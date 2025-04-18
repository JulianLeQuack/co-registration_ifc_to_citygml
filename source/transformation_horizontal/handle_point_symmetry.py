import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def check_point_symmetry(features: np.array, tolerance=1e-6) -> bool:
    # Extract just the point coordinates from the features array (columns 2 and 3 contain x,y)
    points = features[:, 2:4]
    
    # Calculate centroid directly from the points
    centroid = np.mean(points, axis=0)
    reflected_points = 2 * centroid - points

    # Find nearest neighbors
    tree = cKDTree(points)
    distances, _ = tree.query(reflected_points, k=1)

    # return True if points create a point-symmetric shape
    return np.all(distances < tolerance)



if __name__ == "__main__":
    from source.transformation_horizontal.create_footprints.create_DXF_footprint_polygon import create_DXF_footprint_polygon
    from source.transformation_horizontal.detect_features import detect_features, filter_features_by_feature_triangle_area

    #Example points
    input_path = "./demo/data/01-05-0507_EG.1.dxf"
    layer_name = "A_09_TRAGDECKE"
    dxf_footprint = create_DXF_footprint_polygon(dxf_path=input_path, layer_name=layer_name)
    dxf_features = detect_features(footprint=dxf_footprint, angle_threshold_deg=30)
    dxf_features_filtered = filter_features_by_feature_triangle_area(features=dxf_features, min_area=5)
    dxf_features_filtered_sym = filter_features_by_feature_triangle_area(features=dxf_features, min_area=10)

    plt.figure(figsize=(10,10))
    plt.plot(dxf_features_filtered[:, 2], dxf_features_filtered[:, 3], 
                color="blue", 
                label=f"Point-symmetric: {check_point_symmetry(dxf_features_filtered)}")
    plt.plot(dxf_features_filtered_sym[:, 2], dxf_features_filtered_sym[:, 3], 
                color="red", 
                label=f"Point-symmetric: {check_point_symmetry(dxf_features_filtered_sym)}")
    plt.grid(True)
    plt.legend()
    plt.show()