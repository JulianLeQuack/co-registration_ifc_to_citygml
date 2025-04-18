import numpy as np
import matplotlib.pyplot as plt
import itertools
from shapely.geometry import Polygon, MultiPolygon

from source.transformation_horizontal.create_footprints.create_CityGML_footprint import create_CityGML_footprint
from source.transformation_horizontal.detect_features import detect_features, filter_features_by_edge_length, filter_features_by_feature_triangle_area
from source.transformation_horizontal.create_footprints.create_DXF_footprint_polygon import create_DXF_footprint_polygon
from source.transformation_horizontal.create_footprints.create_IFC_footprint_polygon import create_IFC_footprint_polygon

from source.transformation_horizontal.rigid_transformation import Rigid_Transformation


def estimate_transformation_from_2pairs(source1, source2, target1, target2) -> Rigid_Transformation:
    """
    Estimate the 2D rigid transformation (rotation and translation) from two pairs of corresponding points.
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


def estimate_rigid_transformation(source_features, target_features, distance_tol=5.0, angle_tol_deg=10.0):
    """
    Use all possible feature combinations to estimate the best rigid transformation that aligns source_features to target_features.
    
    Parameters:
      source_features, target_features: numpy arrays with shape (n,5) where each row is
                           [poly_index, vertex_index, x_coordinate, y_coordinate, turning_angle_deg]
      distance_tol: Maximum allowed distance for a transformed feature to be considered an inlier.
      angle_tol_deg: Maximum allowed difference in turning angle (degrees) for a candidate match.
      
    Returns:
      best_transformation: A Rigid_Transformation object representing the best transformation.
      best_inliers: List of inlier feature pairs ((feature from source, matching feature from target)).
    """
    best_inlier_count = 0
    best_transformation = None
    best_inliers = []
    angle_tol = np.radians(angle_tol_deg)
    
    if len(source_features) < 2 or len(target_features) < 2:
        print("Not enough features for estimation.")
        return None, []
    
    # Loop over all pairs of features from the first set.
    for f1_1, f1_2 in itertools.combinations(source_features, 2):
        # Use columns 2 and 3 as point coordinates.
        p1, p2 = np.array(f1_1[2:4]), np.array(f1_2[2:4])
        # Use column 4 as turning angle in degrees (converted to radians).
        a1, a2 = np.radians(f1_1[4]), np.radians(f1_2[4])
        
        # Find candidate matches in target_features based on similar turning angles.
        candidates1 = [f for f in target_features if abs(np.radians(f[4]) - a1) < angle_tol]
        candidates2 = [f for f in target_features if abs(np.radians(f[4]) - a2) < angle_tol]
        
        if not candidates1 or not candidates2:
            continue
        
        # Loop over candidate pairs from target_features.
        for f2_1 in candidates1:
            for f2_2 in candidates2:
                if np.array_equal(f2_1, f2_2):
                    continue  # Skip if the same candidate is used twice.
                q1, q2 = np.array(f2_1[2:4]), np.array(f2_2[2:4])
                    
                # Avoid degenerate cases.
                if np.linalg.norm(p2 - p1) < 1e-3 or np.linalg.norm(q2 - q1) < 1e-3:
                    continue
                
                # Estimate the candidate transformation using the two point pairs.
                candidate_transformation = estimate_transformation_from_2pairs(p1, p2, q1, q2)
                
                # Evaluate inliers: transform each feature in source_features and look for a corresponding match in target_features.
                inliers = []
                for f in source_features:
                    pt = np.array(f[2:4])
                    ang = f[4]  # in degrees
                    pt_trans = candidate_transformation.rotation_matrix() @ pt + candidate_transformation.translation_vector()
                    best_match = None
                    for g in target_features:
                        pt2 = np.array(g[2:4])
                        ang2 = g[4]
                        if (np.linalg.norm(pt_trans - pt2) < distance_tol and 
                            abs(np.radians(ang) - np.radians(ang2)) < angle_tol):
                            best_match = g
                            break
                    if best_match is not None:
                        inliers.append((f, best_match))
                
                if len(inliers) > best_inlier_count:
                    best_inlier_count = len(inliers)
                    best_transformation = candidate_transformation
                    best_inliers = inliers
                    
    return best_transformation, best_inliers


def refine_rigid_transformation(inlier_pairs):
    """
    Refine the rigid transformation parameters (translation and rotation) using least squares.
    """
    if len(inlier_pairs) < 2:
        print("Not enough inlier pairs for refinement. Returning None.")
        return None

    # Extract corresponding points (x and y) from the inlier pairs.
    P = np.array([pair[0][2:4] for pair in inlier_pairs])  # Points from source.
    Q = np.array([pair[1][2:4] for pair in inlier_pairs])  # Corresponding points from target.
    
    # Compute centroids.
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    # Center the points.
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # Compute the cross-covariance matrix.
    H = P_centered.T @ Q_centered
    
    # Compute SVD of H.
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure a proper rotation (determinant = 1).
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute the translation vector.
    t = centroid_Q - R @ centroid_P
    
    # Extract the rotation angle.
    theta = np.arctan2(R[1, 0], R[0, 0])
    
    # Create the refined transformation object.
    refined_transformation = Rigid_Transformation(t=t, theta=theta)
    
    return refined_transformation


def main():
    # Define file paths and other settings.
    # ifc_path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"
    ifc_path = "./test_data/ifc/3.003 01-05-0507_EG.ifc"
    # dxf_path = "./test_data/dxf/01-05-0501_EG.dxf"
    dxf_path = "./test_data/dxf/01-05-0507_EG.1.dxf"
    layer_name = "A_09_TRAGDECKE"  # Update if different
    citygml_path = "./test_data/citygml/TUM_LoD2_Full_withSurrounds.gml"
    # citygml_buildings = ["DEBY_LOD2_4959457"]
    citygml_buildings = ['DEBY_LOD2_4959793', 'DEBY_LOD2_4959323', 'DEBY_LOD2_4959321', 'DEBY_LOD2_4959324', 'DEBY_LOD2_4959459', 'DEBY_LOD2_4959322', 'DEBY_LOD2_4959458']

    # Create footprints.
    polygon_ifc = create_IFC_footprint_polygon(ifc_path)
    polygon_citygml = create_CityGML_footprint(citygml_path, citygml_buildings)
    polygon_dxf = create_DXF_footprint_polygon(dxf_path, layer_name)

    # Detect features (corners) using turning angles.
    features_ifc = detect_features(polygon_ifc, angle_threshold_deg=30)
    features_citygml = detect_features(polygon_citygml, angle_threshold_deg=30)
    features_dxf = detect_features(polygon_dxf, angle_threshold_deg=30)

    # Filter features based on a minimum edge length.
    min_area = 5
    features_ifc_filtered = filter_features_by_feature_triangle_area(features_ifc, min_area=min_area)
    features_citygml_filtered = filter_features_by_feature_triangle_area(features_citygml, min_area=min_area)
    features_dxf_filtered = filter_features_by_feature_triangle_area(features_dxf, min_area=min_area) # need more detail for correct orientation

    print(f"IFC Source: {len(features_ifc)} features, filtered down to {len(features_ifc_filtered)}")
    print(f"CityGML Target: {len(features_citygml)} features, filtered down to {len(features_citygml_filtered)}")
    print(f"DXF Source: {len(features_dxf)} features, filtered down to {len(features_dxf_filtered)}")
    
    # --- Estimate and refine transformation for IFC -> CityGML ---
    print("\nEstimating transformation for IFC to CityGML...")
    rigid_transformation_ifc, inlier_pairs_ifc = estimate_rigid_transformation(
        features_ifc_filtered, features_citygml_filtered, distance_tol=1, angle_tol_deg=45)
    if rigid_transformation_ifc is None:
        print("Estimation for IFC failed to find a valid transformation.")
        return
    print(f"IFC Initial Transformation: theta = {rigid_transformation_ifc.theta}, t = {rigid_transformation_ifc.t}")
    print(f"Number of inlier pairs (IFC): {len(inlier_pairs_ifc)}")
    
    refined_transformation_ifc = refine_rigid_transformation(inlier_pairs_ifc)
    if refined_transformation_ifc is None:
        print("Refinement for IFC failed.")
        return
    print(f"Refined IFC Transformation: theta = {refined_transformation_ifc.theta}, t = {refined_transformation_ifc.t}")
    
    # Apply the refined transformation to the IFC footprint.
    polygon_ifc_transformed = refined_transformation_ifc.transform_shapely_polygon(polygon_ifc)
    
    # --- Register DXF to the transformed IFC instead of CityGML ---
    print("\nEstimating transformation for DXF to transformed IFC...")
    # 1. detect features on the transformed IFC
    features_ifc_transformed = detect_features(polygon_ifc_transformed, angle_threshold_deg=30)
    features_ifc_transformed_filtered = filter_features_by_feature_triangle_area(
        features_ifc_transformed, min_area=min_area
    )
    # 2. estimate & refine DXF -> transformed IFC
    rigid_transformation_dxf, inlier_pairs_dxf = estimate_rigid_transformation(
        features_dxf_filtered, features_ifc_transformed_filtered, distance_tol=1, angle_tol_deg=45
    )
    if rigid_transformation_dxf is None:
        print("Estimation for DXF failed to find a valid transformation.")
        return

    # Print initial DXF→IFC transformation and inlier count
    print(f"\nDXF Initial Transformation: theta = {rigid_transformation_dxf.theta}, t = {rigid_transformation_dxf.t}")
    print(f"Number of inlier pairs (DXF to IFC): {len(inlier_pairs_dxf)}")

    refined_transformation_dxf = refine_rigid_transformation(inlier_pairs_dxf)
    if refined_transformation_dxf is None:
        print("Refinement for DXF failed.")
        return

    # Print refined DXF→IFC transformation
    print(f"Refined DXF Transformation: theta = {refined_transformation_dxf.theta}, t = {refined_transformation_dxf.t}")

    polygon_dxf_transformed = refined_transformation_dxf.transform_shapely_polygon(polygon_dxf)

    # --- Plot all three footprints (aligned) ---
    plt.figure(figsize=(10, 10))
    
    # Plot CityGML footprint (target) in blue.
    if polygon_citygml.geom_type == "MultiPolygon":
        for i, poly in enumerate(polygon_citygml.geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'b-', linewidth=2, label='CityGML Footprint' if i == 0 else "")
    elif polygon_citygml.geom_type == "Polygon":
        x, y = polygon_citygml.exterior.xy
        plt.plot(x, y, 'b-', linewidth=2, label='CityGML Footprint')

    # Plot transformed IFC footprint in red.
    if polygon_ifc_transformed.geom_type == "MultiPolygon":
        for i, poly in enumerate(polygon_ifc_transformed.geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'r-', linewidth=2, label='Transformed IFC Footprint' if i == 0 else "")
    elif polygon_ifc_transformed.geom_type == "Polygon":
        x, y = polygon_ifc_transformed.exterior.xy
        plt.plot(x, y, 'r-', linewidth=2, label='Transformed IFC Footprint')
    
    # Plot transformed DXF footprint in green.
    if polygon_dxf_transformed.geom_type == "MultiPolygon":
        for i, poly in enumerate(polygon_dxf_transformed.geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'g-', linewidth=2, label='Transformed DXF Footprint' if i == 0 else "")
    elif polygon_dxf_transformed.geom_type == "Polygon":
        x, y = polygon_dxf_transformed.exterior.xy
        plt.plot(x, y, 'g-', linewidth=2, label='Transformed DXF Footprint')
    
    plt.axis('equal')
    plt.title("Coregistered Footprints: IFC, CityGML, and DXF")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
