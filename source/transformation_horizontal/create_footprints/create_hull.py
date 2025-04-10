import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import MultiPoint, LineString, MultiLineString, MultiPolygon
from shapely.ops import substring, unary_union

from alphashape import alphashape


# def create_convex_hull(points: np.array):
#     # Create a MultiPoint object and compute its convex hull
#     footprint_points = MultiPoint(points)
#     footprint_convex_hull = footprint_points.convex_hull.boundary

#     # Sample many points along the boundary for alignment
#     footprint_convex_hull_densified = MultiPoint()
#     for i in np.arange(0, footprint_convex_hull.length, 0.2):
#         s = substring(footprint_convex_hull, i, i+0.2)
#         footprint_convex_hull_densified = footprint_convex_hull_densified.union(s.boundary)

#     # Create np array from MultiPoint object
#     result = np.array([(point.x, point.y) for point in footprint_convex_hull_densified.geoms])
#     return result


def create_hull(points: np.array, alpha: float = 0.2) -> MultiPolygon:
    """
    Creates a concave hull from a set of points, always returned as a MultiPolygon.
    If the alphashape result is a single Polygon, it's wrapped in a MultiPolygon.
    If the alphashape returns a MultiPolygon, it is returned as-is.
    For other geometry types or an empty result, an empty MultiPolygon is returned.
    """
    footprint_alphashape = alphashape(points, alpha)

    if footprint_alphashape.is_empty:
        return MultiPolygon([])

    geom_type = footprint_alphashape.geom_type
    if geom_type == 'Polygon':
        return MultiPolygon([footprint_alphashape])
    elif geom_type == 'MultiPolygon':
        return footprint_alphashape
    else:
        # In case the alphashape returns a geometry we don't expect (e.g. LineString),
        # we return an empty MultiPolygon.
        return MultiPolygon([])


if __name__ == "__main__":
    from source.transformation_horizontal.create_footprints.create_CityGML_footprint import create_CityGML_footprint
    from source.transformation_horizontal.create_footprints.create_IFC_footprint import create_IFC_footprint

    # Set alpha values for alphashape
    alpha_ifc = 0.1
    alpha_citygml = 0

    # Create footprints for IFC
    ifc_original = create_IFC_footprint("./test_data/ifc/3.003 01-05-0507_EG.ifc")
    # ifc_convex = create_convex_hull(ifc_original)
    ifc_concave = create_hull(ifc_original, alpha_ifc)
    print(f"Concave Hull Type: {type(ifc_concave)}")

    # Create footprints for CityGML
    # citygml_original = create_CityGML_footprint("./test_data/citygml/TUM_LoD2_Full_withSurrounds.gml", ["DEBY_LOD2_4959457"])
    # citygml_convex = create_convex_hull(citygml_original)
    # citygml_concave = create_concave_hull(citygml_original, alpha_citygml)

    plt.figure(figsize=(18, 12))

    # IFC Original
    plt.subplot(2, 3, 1)
    if ifc_original.size > 0:
        plt.scatter(ifc_original[:, 0], ifc_original[:, 1], c='k',
                    label='IFC Original', s=5)
    plt.title('IFC Original')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()

    # # IFC Convex Hull
    # plt.subplot(2, 3, 2)
    # if ifc_convex.size > 0:
    #     plt.plot(ifc_convex[:, 0], ifc_convex[:, 1], c='b',
    #              label='IFC Convex Hull')
    # plt.title('IFC Convex Hull')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.grid(True)
    # plt.legend()

    # IFC Concave Hull
    plt.subplot(2, 3, 3)
    if not ifc_concave.is_empty:
        for poly in ifc_concave.geoms:
            x, y = poly.exterior.xy
            plt.plot(x, y, c='r', label=f'IFC Concave Hull (α={alpha_ifc})')
    else:
        plt.text(0.5, 0.5, "Empty", horizontalalignment='center')
    plt.title(f'IFC Concave Hull (α={alpha_ifc})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()

    # # CityGML Original
    # plt.subplot(2, 3, 4)
    # if citygml_original.size > 0:
    #     plt.scatter(citygml_original[:, 0], citygml_original[:, 1], c='k',
    #                 label='CityGML Original', s=5)
    # plt.title('CityGML Original')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.grid(True)
    # plt.legend()

    # # CityGML Convex Hull
    # plt.subplot(2, 3, 5)
    # if citygml_convex.size > 0:
    #     plt.plot(citygml_convex[:, 0], citygml_convex[:, 1], c='g',
    #              label='CityGML Convex Hull')
    # plt.title('CityGML Convex Hull')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.grid(True)
    # plt.legend()

    # # CityGML Concave Hull
    # plt.subplot(2, 3, 6)
    # if not citygml_concave.is_empty:
    #     for poly in citygml_concave.geoms:
    #         x, y = poly.exterior.xy
    #         plt.plot(x, y, c='m',
    #                  label=f'CityGML Concave Hull (α={alpha_citygml})')
    # else:
    #     plt.text(0.5, 0.5, "Empty", horizontalalignment='center')
    # plt.title(f'CityGML Concave Hull (α={alpha_citygml})')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.grid(True)
    # plt.legend()

    plt.tight_layout()
    plt.show()