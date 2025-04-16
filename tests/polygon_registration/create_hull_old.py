import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import MultiPoint, LineString, MultiLineString
from shapely.ops import substring, unary_union

from alphashape import alphashape


def create_convex_hull(points: np.array):
    # Create a MultiPoint object and compute its convex hull
    footprint_points = MultiPoint(points)
    footprint_convex_hull = footprint_points.convex_hull.boundary

    # Sample many points along the boundary for alignment
    footprint_convex_hull_densified = MultiPoint()
    for i in np.arange(0, footprint_convex_hull.length, 0.2):
        s = substring(footprint_convex_hull, i, i+0.2)
        footprint_convex_hull_densified = footprint_convex_hull_densified.union(s.boundary)

    #Create np array from Multipoint object
    result = np.array([(point.x,point.y) for point in footprint_convex_hull_densified.geoms])

    return result

def create_concave_hull(points: np.array, alpha: float=0.2):
    """
    Creates a concave hull from a set of points, handling MultiLineString results.
    """
    footprint_points = points
    footprint_alphashape = alphashape(footprint_points, alpha)

    footprint = footprint_alphashape.exterior.coords

    # densified_lines = []  # Store densified LineStrings

    # if isinstance(footprint_alphashape, LineString):
    #     # Handle LineString directly
    #     for i in np.arange(0, footprint_alphashape.length, 0.2):
    #         s = substring(footprint_alphashape, i, i + 0.2)
    #         densified_lines.append(s)

    # elif isinstance(footprint_alphashape, MultiLineString):
    #     # Handle MultiLineString by iterating through components
    #     for line in footprint_alphashape.geoms:
    #         for i in np.arange(0, line.length, 0.2):
    #             s = substring(line, i, i + 0.2)
    #             densified_lines.append(s)
    # elif footprint_alphashape.is_empty:
    #     return np.array([]) #Return empty array if alphashape is empty.
    # else:
    #     # Handle other geometry types (if necessary)
    #     return np.array([])  # Return empty array or handle as needed

    # # Combine densified LineStrings
    # if densified_lines:
    #     densified_multiline = unary_union(densified_lines)
    #     if isinstance(densified_multiline, LineString):
    #         points_list = [(x,y) for x,y in densified_multiline.coords]
    #         return np.array(points_list)
    #     elif isinstance(densified_multiline, MultiLineString):
    #         points_list = []
    #         for line in densified_multiline.geoms:
    #             points_list.extend([(x,y) for x,y in line.coords])
    #         return np.array(points_list)

    #     else:
    #         return np.array([])#return empty array if unary_union fails.

    # else:
    #     return np.array([])  # Return empty array if no densified lines

    return np.array(footprint)


if __name__ == "__main__":
    from source.transformation_horizontal.create_footprints.create_CityGML_footprint import create_CityGML_footprint
    from source.transformation_horizontal.create_footprints.create_IFC_footprint import create_IFC_footprint


    alpha_ifc = 0.2
    alpha_citygml = 0.05

    ifc_original = create_IFC_footprint("./test_data/ifc/3.002 01-05-0501_EG.ifc")
    ifc_convex_hull = create_convex_hull(ifc_original)
    ifc_alphashape = create_concave_hull(ifc_original, alpha_ifc)

    citygml_original = create_CityGML_footprint("./test_data/citygml/DEBY_LOD2_4959457.gml")
    citygml_convex_hull = create_convex_hull(citygml_original)
    citygml_alphashape = create_concave_hull(citygml_original, alpha_citygml)

    print(ifc_original[0])
    print(ifc_convex_hull[0])
    print(ifc_alphashape[0])

    print(citygml_original[0])
    print(citygml_convex_hull[0])
    print(citygml_alphashape[0])

    plt.figure(figsize=(18, 12))

    # IFC (Scatter plot)
    plt.subplot(2, 3, 1)
    if ifc_original.size > 0:
        plt.scatter(ifc_original[:, 0], ifc_original[:, 1], c='k', label='IFC Original', s=5) #s is marker size
    plt.title('IFC Original')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()

    # IFC Convex Hull (Scatter plot)
    plt.subplot(2, 3, 2)
    if ifc_convex_hull.size > 0:
        plt.scatter(ifc_convex_hull[:, 0], ifc_convex_hull[:, 1], c='b', label='IFC Convex Hull', s=5) #s is marker size
    plt.title('IFC Convex Hull')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()

    # IFC Alpha Shape (Scatter plot)
    plt.subplot(2, 3, 3)
    if ifc_alphashape.size > 0:
        plt.scatter(ifc_alphashape[:, 0], ifc_alphashape[:, 1], c='r', label=f'IFC Alpha Shape (α={alpha_ifc})', s=5)
    plt.title(f'IFC Alpha Shape (α={alpha_ifc})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()

    # CityGML (Scatter plot)
    plt.subplot(2, 3, 4)
    if citygml_original.size > 0:
        plt.scatter(citygml_original[:, 0], citygml_original[:, 1], c='k', label='CityGML Original', s=5) #s is marker size
    plt.title('CityGML Original')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()

    # CityGML Convex Hull (Scatter plot)
    plt.subplot(2, 3, 5)
    if citygml_convex_hull.size > 0:
        plt.scatter(citygml_convex_hull[:, 0], citygml_convex_hull[:, 1], c='g', label='CityGML Convex Hull', s=5)
    plt.title('CityGML Convex Hull')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()

    # CityGML Alpha Shape (Scatter plot)
    plt.subplot(2, 3, 6)
    if citygml_alphashape.size > 0:
        plt.scatter(citygml_alphashape[:, 0], citygml_alphashape[:, 1], c='m', label=f'CityGML Alpha Shape (α={alpha_citygml})', s=5)
    plt.title(f'CityGML Alpha Shape (α={alpha_citygml})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()