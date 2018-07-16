import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas

def array_to_raster(array, pixel_size, x_min, y_max, wkt, path):
    """Writes a GeoTIFF raster from a numpy array.

    :param array: 2D numpy array of cell values
    :param pixel_size: -- Desired resolution of the output raster, in same units as wkt projection.
    :param x_min: Minimum x coordinate (top left corner of raster)
    :param y_max: Maximum y coordinate
    :param wkt: The wkt string with desired projection
    :param path: The output bath of the GeoTIFF
    """
    # First flip the array
    array = np.flipud(array)

    transform = rasterio.transform.from_origin(x_min, y_max, pixel_size, pixel_size)
    out_dataset = rasterio.open(path, 'w', driver='GTiff', height=array.shape[0], width = array.shape[1], count=1,
                                dtype=str(array.dtype),crs=wkt, transform=transform)

    out_dataset.write(array, 1)
    out_dataset.close()

def array_to_polygons(array, affine=None):
    """
    Returns a geopandas dataframe of polygons as deduced from an array.

    :param array: The 2D numpy array to polygonize.
    :param affine: The affine transformation.
    :return:
    """
    if affine == None:
        results = [
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v)
                in enumerate(shapes(array))
        ]
    else:
        results = [
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v)
            in enumerate(shapes(array, transform=affine))
        ]


    tops_df = geopandas.GeoDataFrame({'geometry': [shape(results[geom]['geometry']) for geom in range(len(results))],
                                      'raster_val': [results[geom]['properties']['raster_val'] for geom in range(len(results))]})

    return(tops_df)
