import numpy as np
import matplotlib.pyplot as plt
import itertools
from shapely.geometry import Polygon, MultiPolygon

from source.transformation_horizontal.create_footprints.create_CityGML_footprint import create_CityGML_footprint
from source.transformation_horizontal.detect_features import detect_features, filter_features_by_edge_length
from source.transformation_horizontal.create_footprints.create_DXF_footprint_polygon import create_DXF_footprint_polygon
from source.transformation_horizontal.create_footprints.create_IFC_footprint_polygon import create_IFC_footprint_polygon

from source.transformation_horizontal.rigid_transformation import Rigid_Transformation



def estimate_transformation_from_2pairs(source1, source2, target1, target2) -> Rigid_Transformation:
    """
    Estimate the 2D rigid transformation (rotation and translation) from two pairs of corresponding points.

    Parameters:
        source1, source2: Two 2D points from the source shape (e.g., polygon).
        target1, target2: Corresponding 2D points in the target shape.

    Returns:
        A new Rigid_Transformation object representing the estimated transformation.
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

def estimate_rigid_transformation(features1, features2, distance_tol=5.0, angle_tol_deg=10.0):
    """
    Use all possible feature combinations to estimate the best rigid transformation that aligns features1 to features2.
    
    Parameters:
      features1, features2: numpy arrays with shape (n,5) where each row is
                           [poly_index, vertex_index, x_coordinate, y_coordinate, turning_angle_deg]
      distance_tol: Maximum allowed distance for a transformed feature to be considered an inlier.
      angle_tol_deg: Maximum allowed difference in turning angle (degrees) for a candidate match.
      
    Returns:
      best_transformation: A Rigid_Transformation object representing the best transformation.
      best_inliers: List of inlier feature pairs ((feature from polygon1, matching feature from polygon2)).
    """
    best_inlier_count = 0
    best_transformation = None
    best_inliers = []
    angle_tol = np.radians(angle_tol_deg)
    
    if len(features1) < 2 or len(features2) < 2:
        print("Not enough features for estimation.")
        return None, []
    
    # Loop over all pairs of features from the first set.
    for f1_1, f1_2 in itertools.combinations(features1, 2):
        # Use columns 2 and 3 as point coordinates.
        p1, p2 = np.array(f1_1[2:4]), np.array(f1_2[2:4])
        # Use column 4 as turning angle in degrees (converted to radians).
        a1, a2 = np.radians(f1_1[4]), np.radians(f1_2[4])
        
        # Find candidate matches in features2 based on similar turning angles.
        candidates1 = [f for f in features2 if abs(np.radians(f[4]) - a1) < angle_tol]
        candidates2 = [f for f in features2 if abs(np.radians(f[4]) - a2) < angle_tol]
        
        if not candidates1 or not candidates2:
            continue
        
        # Loop over candidate pairs from features2.
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
                
                # Evaluate inliers: transform each feature in features1 and look for a corresponding match in features2.
                inliers = []
                for f in features1:
                    pt = np.array(f[2:4])
                    ang = f[4]  # in degrees
                    pt_trans = candidate_transformation.rotation_matrix() @ pt + candidate_transformation.translation_vector()
                    best_match = None
                    for g in features2:
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
    
    Parameters:
      inlier_pairs: List of tuples ((feature from polygon1, matching feature from polygon2))
                    where each feature is an array/list of 5 numbers:
                    [poly_index, vertex_index, x_coordinate, y_coordinate, turning_angle_deg].
                    
    Returns:
      refined_transformation: A refined Rigid_Transformation object.
    """
    if len(inlier_pairs) < 2:
        print("Not enough inlier pairs for refinement. Returning None.")
        return None

    # Extract corresponding points (x and y) from the inlier pairs.
    P = np.array([pair[0][2:4] for pair in inlier_pairs])  # Points from polygon1.
    Q = np.array([pair[1][2:4] for pair in inlier_pairs])  # Corresponding points from polygon2.
    
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
    # ifc_path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"
    ifc_path = "./test_data/ifc/3.003 01-05-0507_EG.ifc"

    dxf_path = "./test_data/dxf/01-05-0501_EG.dxf"
    layer_name = "A_09_TRAGDECKE"  # Update if different

    citygml_path = "./test_data/citygml/TUM_LoD2_Full_withSurrounds.gml"
    citygml_buildings = ['DEBY_LOD2_4959793', 'DEBY_LOD2_4959323', 'DEBY_LOD2_4959321',
                         'DEBY_LOD2_4959324', 'DEBY_LOD2_4959459', 'DEBY_LOD2_4959322',
                         'DEBY_LOD2_4959458']
    # citygml_buildings = ["DEBY_LOD2_4959457"]

    # Create the two example footprints.
    polygon_ifc = create_IFC_footprint_polygon(ifc_path)
    polygon_citygml = create_CityGML_footprint(citygml_path, citygml_buildings)
    polygon_dxf = create_DXF_footprint_polygon(dxf_path, layer_name)

    # Detect features (corners) using turning angles.
    features_ifc = detect_features(polygon_ifc, angle_threshold_deg=45)
    features_ifc = detect_features(polygon_citygml, angle_threshold_deg=45)
    features_dxf = detect_features(polygon_dxf, angle_threshold_deg=45)

    features_ifc_filtered = filter_features_by_edge_length(features_ifc, polygon_ifc, min_edge_len=7.0)
    features_ifc_filtered = filter_features_by_edge_length(features_ifc, polygon_citygml, min_edge_len=7.0)
    features_dxf_filtered = filter_features_by_edge_length(features_dxf, polygon_dxf, min_edge_len=7.0)

    print(f"IFC Source: {len(features_ifc)} features, filtered down to {len(features_ifc_filtered)}")
    print(f"CityGML Target: {len(features_ifc)} features, filtered down to {len(features_ifc_filtered)}")
    print(f"DXF Source: {len(features_dxf)} features, filtered down to {len(features_dxf_filtered)}")
    
    # Rough estimation.
    rigid_transformation, inlier_pairs = estimate_rigid_transformation(features_ifc_filtered, features_ifc_filtered,
                                                 distance_tol=1,
                                                 angle_tol_deg=45)
    if rigid_transformation is None:
        print("Estimation failed to find a valid transformation.")
        return
    
    print("\nBest transformation found (initial):")
    print(f"Rotation angle theta [radians]: {rigid_transformation.theta}")
    print(f"Translation vector t [x, y]: {rigid_transformation.t}")
    print("Number of inlier pairs: ", len(inlier_pairs))
    
    # Refine the transformation.
    refined_transformation = refine_rigid_transformation(inlier_pairs)
    if refined_transformation is None:
        print("Refinement failed.")
        return

    print("\nRefined transformation:")
    print(f"Rotation angle theta [radians]: {refined_transformation.theta}")
    print(f"Translation vector t [x, y]: {refined_transformation.t}")
    
    # Apply the refined transformation to the entire multiPolygon footprint (polygon_ifc).
    # Build a new MultiPolygon with each polygon transformed.
    transformed_polys = []
    if hasattr(polygon_ifc, "geom_type"):
        if polygon_ifc.geom_type == "MultiPolygon":
            for poly in polygon_ifc.geoms:
                # Get full exterior
                coords = np.array(poly.exterior.coords)
                transformed_coords = refined_transformation.apply_transformation(coords)
                # Ensure closure: if first and last point are not the same, append the first.
                if not np.allclose(transformed_coords[0], transformed_coords[-1]):
                    transformed_coords = np.vstack([transformed_coords, transformed_coords[0]])
                transformed_polys.append(Polygon(transformed_coords))
            polygon_ifc_transformed = MultiPolygon(transformed_polys)
        elif polygon_ifc.geom_type == "Polygon":
            coords = np.array(polygon_ifc.exterior.coords)
            transformed_coords = refined_transformation.apply_transformation(coords)
            if not np.allclose(transformed_coords[0], transformed_coords[-1]):
                transformed_coords = np.vstack([transformed_coords, transformed_coords[0]])
            polygon_ifc_transformed = Polygon(transformed_coords)
        else:
            print("Unexpected geometry type for polygon_ifc.")
            return
    else:
        print("polygon_ifc is not a Shapely geometry.")
        return

    # Similarly, ensure polygon_citygml is handled for plotting. Assume polygon_citygml is already a Shapely geometry.
    # Plot both footprints.
    plt.figure(figsize=(10, 10))
    
    # Plot transformed IFC footprint (polygon_ifc_transformed)
    if polygon_ifc_transformed.geom_type == "MultiPolygon":
        for poly in polygon_ifc_transformed.geoms:
            x, y = poly.exterior.xy
            plt.plot(x, y, 'r-', linewidth=2, label='Transformed IFC Footprint')
    elif polygon_ifc_transformed.geom_type == "Polygon":
        x, y = polygon_ifc_transformed.exterior.xy
        plt.plot(x, y, 'r-', linewidth=2, label='Transformed IFC Footprint')
    
    # Plot CityGML footprint (polygon_citygml). If MultiPolygon, iterate.
    if polygon_citygml.geom_type == "MultiPolygon":
        for poly in polygon_citygml.geoms:
            x, y = poly.exterior.xy
            plt.plot(x, y, 'b-', linewidth=2, label='CityGML Footprint')
    elif polygon_citygml.geom_type == "Polygon":
        x, y = polygon_citygml.exterior.xy
        plt.plot(x, y, 'b-', linewidth=2, label='CityGML Footprint')
    
    plt.axis('equal')
    plt.title("Footprints with Refined Transformation Applied")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
