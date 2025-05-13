import argparse
import ezdxf
from shapely.geometry import LineString, Polygon, MultiPolygon, mapping
from shapely.ops import unary_union, polygonize
import matplotlib.pyplot as plt

def to_2d(pt):
    return (pt[0], pt[1])

def export_detailed_dxf_footprint(dxf_path: str, layer_name: str, use_origin_filter: bool = True, origin_threshold: float = 10.0) -> MultiPolygon:
    """
    Extracts all line segments from the given DXF layer and polygonizes them.
    Returns a MultiPolygon of the raw polygons (no simplification, no exterior-only),
    then plots them.
    """
    # Read DXF and get entities on the layer
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    ents = msp.query(f'*[layer=="{layer_name}"]')

    # Collect all 2D segments
    segments = []
    for ent in ents:
        try:
            for geom in ent.virtual_entities():
                if geom.dxftype() == "LINE":
                    s = to_2d(geom.dxf.start)
                    e = to_2d(geom.dxf.end)
                    segments.append(LineString([s, e]))
                elif geom.dxftype() in ("POLYLINE", "LWPOLYLINE"):
                    pts = [to_2d(p) for p in geom.points()]
                    for i in range(len(pts) - 1):
                        segments.append(LineString([pts[i], pts[i + 1]]))
                else:
                    for sub in geom.virtual_entities():
                        if sub.dxftype() == "LINE":
                            s = to_2d(sub.dxf.start)
                            e = to_2d(sub.dxf.end)
                            segments.append(LineString([s, e]))
        except Exception:
            continue

    # Filter out segments that start or end at (0,0) if their length exceeds a threshold
    if use_origin_filter:
        filtered_segments = []
        for seg in segments:
            start, end = seg.coords[0], seg.coords[-1]
            if ((start == (0, 0) or end == (0, 0)) and seg.length > origin_threshold):
                continue
            filtered_segments.append(seg)
        segments = filtered_segments

    # Merge and polygonize
    merged = unary_union(segments)
    polys = list(polygonize(merged))
    if not polys:
        raise ValueError("No polygons formed from the DXF layer.")

    # Keep all polygons as-is
    footprint_mp = MultiPolygon(polys)

    # Plot the detailed polygons
    plt.figure(figsize=(10, 10))
    for poly in footprint_mp.geoms:
        x, y = poly.exterior.xy
        plt.plot(x, y, color="blue", linewidth=1)
        for interior in poly.interiors:
            ix, iy = interior.xy
            plt.plot(ix, iy, color="red", linewidth=1)
    plt.gca().set_aspect("equal", "box")
    plt.title(f"Detailed DXF Footprint: {layer_name}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Export detailed (unsimplified) DXF footprint polygons"
    )
    parser.add_argument("-i", "--dxf",    required=True, help="DXF file path")
    parser.add_argument("-l", "--layer",  required=True, help="DXF layer name")
    parser.add_argument("-o", "--output", default="dxf_footprint_detailed.geojson",
                        help="Output GeoJSON path (ignored when plotting)")
    args = parser.parse_args()
    export_detailed_dxf_footprint(args.dxf, args.layer, args.output)

if __name__ == "__main__":
    main()