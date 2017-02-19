import pandas as pd
import numpy as np

def elev_points(tiff, cloud):
    """Normalizes a point cloud."""
    #TODO: Match tiff and cloud projections.
    import gdal
    import affine

    # A bunch of geospatial information.
    raster_file = tiff
    raster_object = gdal.Open(raster_file)
    raster_array = np.array(raster_object.GetRasterBand(1).ReadAsArray())
    geo_trans = raster_object.GetGeoTransform()
    forward_transform = affine.Affine.from_gdal(*geo_trans)
    reverse_transform = ~forward_transform
    cloud_length = len(cloud.scaled_xy)

    i = 0
    z_list = []

    def retrieve_pixel_value(geo_coord, data_source):
        """Return floating-point value that corresponds to given point."""
        x, y = geo_coord[0], geo_coord[1]
        px, py = reverse_transform * (x, y)
        px, py = int(px + 0.5), int(py + 0.5)
        pixel_coord = px, py

        return raster_array[pixel_coord[1]][pixel_coord[0]]
    # TODO: make this a numpy operation
    for coord in cloud.scaled_xy:
        try:
            if i % 10000 == 0:
                #TODO: Modulo may be slowing this down.
                z_list.append(retrieve_pixel_value(coord, raster_object))
                print(i, "points normalized out of", cloud_length)
                i+=1
            else:
                z_list.append(retrieve_pixel_value(coord, raster_object))
                i+=1
        except IndexError:
            # TODO: consider alternatives to this method.
            # Should be resolved after clipping method is complete.
            z_list.append(0)

    cloud.dataframe['elev'] = z_list
    cloud.dataframe['norm'] = cloud.dataframe['z'] - cloud.dataframe['elev']

    # Some cleaning processes


    cloud.dataframe.dropna(inplace=True)
    # TODO: A very poor way to get rid of "edge errors"
    cloud.dataframe = cloud.dataframe[cloud.dataframe.elev != 0]


def df_to_las(df, out_path, header, zcol='z'):
    """Exports normalized points to new las."""
    import laspy

    outfile = laspy.file.File(out_path, mode="w", header = header)
    outfile.x = df['x']
    outfile.y = df['y']
    outfile.z = df[zcol]
    # outfile.intensity = df['int']
    # outfile.return_num = df['ret']
