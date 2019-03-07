import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas

# This module holds internal functions for GIS processing.

def get_las_crs():
    """
    Attempts to retrive CRS information from an input `laspy.file.File` object.
    :return:
    """
    pass

def project_indices(indices, raster):
    """
    Converts indices of an array (for example, those indices that describe the location of a local maxima) to the
    same space as the input cloud object.

    :param indices: The indices to project, an Nx2 matrix of indices where the first column are the rows (Y) and
    the second column is the columns (X)
    :param raster: An object of type pyfor.rasterizer.Raster
    :return:
    """

    seed_xy = indices[:,1] + (raster._affine[2] / raster._affine[0]), \
              indices[:,0] + (raster._affine[5] - (raster.grid.cloud.data.max[1] - raster.grid.cloud.data.min[1]) /
                              abs(raster._affine[4]))
    seed_xy = np.stack(seed_xy, axis = 1)
    return(seed_xy)

def array_to_raster(array, affine, wkt, path):
    """Writes a GeoTIFF raster from a numpy array.

    :param array: 2D numpy array of cell values
    :param pixel_size: -- Desired resolution of the output raster, in same units as wkt projection.
    :param x_min: Minimum x coordinate (top left corner of raster)
    :param y_max: Maximum y coordinate
    :param wkt: The wkt string with desired projection
    :param path: The output bath of the GeoTIFF
    """
    # First flip the array
    #transform = rasterio.transform.from_origin(x_min, y_max, pixel_size, pixel_size)
    out_dataset = rasterio.open(path, 'w', driver='GTiff', height=array.shape[0], width = array.shape[1], count=1,
                                dtype=str(array.dtype),crs=wkt, transform=affine)
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

def polygons_to_raster(polygons):
    pass
