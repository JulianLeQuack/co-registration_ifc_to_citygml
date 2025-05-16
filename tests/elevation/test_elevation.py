# dc7bcb7aa5d7e0e56670aa5b8da72d5c

import rasterio
from rasterio.merge import merge

def assemble_tiles(tile_paths):
    src_files_to_mosaic = []
    for path in tile_paths:
        src = rasterio.open(path)
        src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })
    nodata = src_files_to_mosaic[0].nodata
    for src in src_files_to_mosaic:
        src.close()
    return mosaic, out_trans, out_meta, nodata

def get_elevation(mosaic, transform, x, y, nodata):
    col, row = ~transform * (x, y)
    col, row = int(col), int(row)
    if row < 0 or row >= mosaic.shape[1] or col < 0 or col >= mosaic.shape[2]:
        return None
    elevation = mosaic[0, row, col]
    if elevation == nodata:
        return None
    return round(float(elevation), 2)

if __name__ == "__main__":
    tile_files = [
        "./test_data/elevation/690_5335.tif",
        "./test_data/elevation/690_5336.tif",
        "./test_data/elevation/691_5335.tif",
        "./test_data/elevation/691_5336.tif",
    ]
    mosaic, transform, meta, nodata = assemble_tiles(tile_files)
    x, y = 691011, 5336027
    elevation = get_elevation(mosaic, transform, x, y, nodata)
    if elevation is None:
        print("Coordinate outside mosaic bounds or no data")
    else:
        print(f"Elevation at ({x}, {y}): {elevation} meters")

