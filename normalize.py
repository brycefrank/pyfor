import pandas as pd
import numpy as np

def elev_points(tiff, cloud):
    """Normalizes a point cloud."""
    #TODO: Match tiff and cloud projections.
    import gdal
    import affine

    xy_array = cloud.dataframe.as_matrix(columns=['x', 'y'])

    # A bunch of geospatial information.
    raster_file = tiff
    raster_object = gdal.Open(raster_file)
    raster_array = np.array(raster_object.GetRasterBand(1).ReadAsArray())
    geo_trans = raster_object.GetGeoTransform()
    forward_transform = affine.Affine.from_gdal(*geo_trans)
    reverse_transform = ~forward_transform
    cloud_length = len(xy_array)

    i = 0
    z_list = []

    def retrieve_pixel_value(geo_coord):
        """Return floating-point value that corresponds to given point."""
        #TODO: this could at least be vectorized since reverse_transform is a constant
        x, y = geo_coord[0], geo_coord[1]
        px, py = reverse_transform * (x, y)
        px, py = int(px + 0.5), int(py + 0.5)
        pixel_coord = px, py

        return raster_array[pixel_coord[1]][pixel_coord[0]]
    #TODO: make this a numpy operation
    #TODO: or make multiprocessing work...
    for coord in xy_array:
        try:
            z_list.append(retrieve_pixel_value(coord))
        except IndexError:
            # TODO: consider alternatives to this method.
            # Should be resolved after clipping method is complete.
            z_list.append(0)


    cloud.dataframe['elev'] = z_list
    cloud.dataframe['norm'] = cloud.dataframe['z'] - cloud.dataframe['elev']

    # Some cleaning processes
    cloud.dataframe.dropna(inplace=True)
    cloud.dataframe = cloud.dataframe[cloud.dataframe.elev != 0]


def df_to_las(df, out_path, header, zcol='z'):
    """Exports normalized points to new las."""
    import laspy

    outfile = laspy.file.File(out_path, mode="w", header = header)
    outfile.x = df['x']
    outfile.y = df['y']
    outfile.z = df[zcol]
    outfile.intensity = df['int']
    outfile.return_num = df['ret']
