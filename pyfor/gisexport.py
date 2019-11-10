import numpy as np
import rasterio

# This module holds internal functions for GIS processing.

def project_indices(indices, raster):
    """
    Converts indices of an array (for example, those indices that describe the location of a local maxima) to the
    same space as the input cloud object.

    :param indices: The indices to project, an Nx2 matrix of indices where the first column are the rows (Y) and
    the second column is the columns (X)
    :param raster: An object of type pyfor.rasterizer.Raster
    :return:
    """

    seed_xy = (
        indices[:, 1] + (raster._affine[2] / raster._affine[0]),
        indices[:, 0]
        + (
            raster._affine[5]
            - (raster.grid.cloud.data.max[1] - raster.grid.cloud.data.min[1])
            / abs(raster._affine[4])
        ),
    )
    seed_xy = np.stack(seed_xy, axis=1)
    return seed_xy


def array_to_raster(array, affine, crs, path):
    """Writes a GeoTIFF raster from a numpy array.

    :param array: 2D numpy array of cell values
    :param affine: The affine transformation.
    :param crs: A rasterio-compatible coordinate reference (e.g. a proj4 string)
    :param path: The output bath of the GeoTIFF
    """
    # First flip the array
    # transform = rasterio.transform.from_origin(x_min, y_max, pixel_size, pixel_size)
    out_dataset = rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=str(array.dtype),
        crs=crs,
        transform=affine,
    )
    out_dataset.write(array, 1)
    out_dataset.close()

