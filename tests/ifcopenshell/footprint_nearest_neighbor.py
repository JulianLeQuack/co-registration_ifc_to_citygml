import numpy as np
import matplotlib.pyplot as plt
import math

from shapely.geometry import MultiPoint, Polygon, MultiPolygon, Point, LineString
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN


def concave_hull_k(points, k):
    """
    Compute a concave hull (for one cluster) using the k-nearest neighbor approach.
    Input:
      points - a list of (x,y) tuples.
      k - the number of neighbors to consider (must be at least 3).
    Returns:
      A list of hull points (vertices), in order.
    """
    if len(points) < 3:
        return points

    pts = points.copy()
    # Starting point: the point with the lowest y (and then lowest x)
    first = min(pts, key=lambda p: (p[1], p[0]))
    hull = [first]
    current = first
    pts.remove(first)
    # initial reference angle (in degrees) pointing to the right (0°)
    current_angle = 0

    while True:
        if not pts:
            break

        k_current = min(k, len(pts))
        # sort remaining points by distance to the current point and take the first k_current
        neighbors = sorted(pts, key=lambda p: math.dist(current, p))[:k_current]

        candidate = None
        candidate_angle = None

        # For each candidate from the k-nearest, compute the angle relative to the previous edge.
        for p in neighbors:
            # Compute angle (in degrees) of vector from current to candidate
            angle = math.degrees(math.atan2(p[1] - current[1], p[0] - current[0]))
            # Adjust relative to the current angle so that we always “turn right”
            angle = (angle - current_angle) % 360
            if candidate is None or angle < candidate_angle:
                candidate = p
                candidate_angle = angle
                candidate_abs_angle = math.degrees(math.atan2(p[1] - current[1], p[0] - current[0]))

        # Check for intersections: if adding the edge from current to candidate creates an intersection with existing hull edges,
        # try the next candidate.
        new_edge = LineString([current, candidate])
        intersect = False
        if len(hull) > 1:
            for i in range(len(hull) - 1):
                seg = LineString([hull[i], hull[i + 1]])
                if new_edge.crosses(seg):
                    intersect = True
                    break
        if intersect:
            found = False
            # Try other candidates to see if one avoids intersection.
            for p in neighbors:
                new_edge = LineString([current, p])
                intersect = False
                for i in range(len(hull) - 1):
                    seg = LineString([hull[i], hull[i + 1]])
                    if new_edge.crosses(seg):
                        intersect = True
                        break
                if not intersect:
                    candidate = p
                    candidate_abs_angle = math.degrees(math.atan2(p[1] - current[1],
                                                                  p[0] - current[0]))
                    found = True
                    break
            if not found:
                break  # no candidate found that avoids intersection

        hull.append(candidate)
        pts.remove(candidate)
        current_angle = candidate_abs_angle
        current = candidate

        if current == first:
            break

    return hull


def compute_concave_hull(points, k=3):
    """
    Compute the concave hull for a set of points (as a NumPy array of shape (n,2))
    using the k-nearest neighbor method.
    If the number of unique points is less than 3, a buffered geometry is created.
    Returns:
      A Shapely Polygon representing the concave hull.
    """
    # Remove duplicates and convert to list of tuples.
    pts = list(set(map(tuple, points)))
    if len(pts) < 3:
        # Too few points to form a valid polygon.
        # Create a buffered MultiPoint as an approximation.
        pts_arr = np.array(pts)
        if pts_arr.shape[0] > 0:
            min_xy = pts_arr.min(axis=0)
            max_xy = pts_arr.max(axis=0)
            diag = math.hypot(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])
            buffer_val = diag * 0.01 if diag > 0 else 1.0
        else:
            buffer_val = 1.0
        return MultiPoint(pts).buffer(buffer_val)

    max_k = len(pts)
    k_current = k
    while k_current <= max_k:
        hull_pts = concave_hull_k(pts, k_current)
        hull_poly = Polygon(hull_pts)
        # Check that every point lies inside or on the hull boundary.
        if all(hull_poly.contains(Point(p)) or hull_poly.touches(Point(p)) for p in pts):
            return hull_poly
        k_current += 1

    # Fallback: return the convex hull if concave hull does not encompass all points.
    return MultiPoint(pts).convex_hull


def cluster_and_concave_hulls(points, k=3, cluster_eps=None, min_samples=3):
    """
    Clusters the input points using DBSCAN and computes a concave hull for each cluster.
    Returns a MultiPolygon of the concave hull(s) for each cluster.
    Parameters:
      points       - a NumPy array of shape (n,2)
      k            - parameter for the k-nearest neighbor concave hull algorithm
      cluster_eps  - DBSCAN’s eps parameter; if None, set to 1% of bounding box diagonal.
      min_samples  - the min_samples parameter for DBSCAN.
    """
    if cluster_eps is None:
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        diag = math.hypot(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])
        cluster_eps = 0.01 * diag

    db = DBSCAN(eps=cluster_eps, min_samples=min_samples)
    labels = db.fit_predict(points)

    hulls = []
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise.
        cluster_pts = points[labels == label]
        poly = compute_concave_hull(cluster_pts, k)
        # Only add non-empty and non-degenerate polygons.
        if not poly.is_empty and poly.area > 0:
            hulls.append(poly)

    if not hulls:
        return MultiPolygon([])
    if len(hulls) == 1:
        return MultiPolygon([hulls[0]])
    return MultiPolygon(hulls)


# ============================
# Main Code (example usage)
# ============================
if __name__ == "__main__":
    # Import your footprint extraction functions.
    from source.transformation_horizontal.create_footprints.create_CityGML_footprint import create_CityGML_footprint
    from source.transformation_horizontal.create_footprints.create_IFC_footprint import create_IFC_footprint

    # Parameter for the k-nearest neighbor concave hull algorithm.
    k_ifc = 3

    # Create footprints for IFC.
    ifc_original = create_IFC_footprint("./test_data/ifc/3.003 01-05-0507_EG.ifc")
    # Compute the concave hull(s) for IFC using the alternative algorithm.
    ifc_concave = cluster_and_concave_hulls(ifc_original, k=k_ifc)

    print(f"Concave Hull Type: {type(ifc_concave)}")

    plt.figure(figsize=(18, 12))

    # Plot IFC Original Points.
    plt.subplot(2, 3, 1)
    if ifc_original.size > 0:
        plt.scatter(ifc_original[:, 0], ifc_original[:, 1], c='k',
                    label='IFC Original', s=5)
    plt.title('IFC Original')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()

    # Plot IFC Concave Hull.
    plt.subplot(2, 3, 3)
    if not ifc_concave.is_empty:
        for poly in ifc_concave.geoms:
            x, y = poly.exterior.xy
            plt.plot(x, y, c='r', label=f'IFC Concave Hull (k={k_ifc})', linewidth=2)
    else:
        plt.text(0.5, 0.5, "Empty", horizontalalignment='center')
    plt.title(f'IFC Concave Hull (k={k_ifc})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
