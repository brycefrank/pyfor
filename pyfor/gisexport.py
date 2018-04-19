import ogr
import osr
import os
import gdal
import numpy as np
import rasterio
from rasterio.features import shapes

def export_wkt_multipoints_to_shp(geom, path):
    if os.path.exists(path):
        os.remove(path)

    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    outDataSource = outDriver.CreateDataSource(path)
    outLayer = outDataSource.CreateLayer(path, geom_type=ogr.wkbMultiPoint)
    featureDefn = outLayer.GetLayerDefn()

    outFeature = ogr.Feature(featureDefn)
    outFeature.SetGeometry(geom)
    outLayer.CreateFeature(outFeature)
    outFeature.Destroy()


def _export_coords_to_shp(coordlist, path):
    """Creates a multipoint shapefile of the coordinates in a 2d array, used for debugging purposes.

    :param coordlist: 2d list of coordinates.
    :param path: Output path of the shapefile.
    """
    if os.path.exists(path):
        os.remove(path)

    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    outDataSource = outDriver.CreateDataSource(path)
    outLayer = outDataSource.CreateLayer(path, geom_type=ogr.wkbMultiPoint)
    featureDefn = outLayer.GetLayerDefn()

    multipoint = ogr.Geometry(ogr.wkbMultiPoint)
    for plot in coordlist:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(plot[0][0], plot[0][1])
        multipoint.AddGeometry(point)

    outFeature = ogr.Feature(featureDefn)
    outFeature.SetGeometry(multipoint)
    outLayer.CreateFeature(outFeature)
    outFeature.Destroy()

def array_to_raster(array, pixel_size, x_min, y_max, wkt, path):
    """Writes a GeoTIFF raster from a numpy array.

    :param array: 2D numpy array of cell values
    :param pixel_size: -- Desired resolution of the output raster, in same units as wkt projection.
    :param x_min: Minimum x coordinate (top left corner of raster)
    :param y_max: Maximum y coordinate
    :param wkt: The wkt string with desired projection
    :param path: The output bath of the GeoTIFF
    """
    transform = rasterio.transform.from_origin(x_min, y_max, pixel_size, pixel_size)
    out_dataset = rasterio.open(path, 'w', driver='GTiff', height=array.shape[0], width = array.shape[1], count=1,
                                dtype=str(array.dtype),crs=wkt, transform=transform)

    out_dataset.write(array, 1)
    out_dataset.close()


def array_to_polygons(array, pixel_size, x_min, y_max, wkt, path):
    """
    Writes a shapefile from a numpy array (calls rasterio.polygonize first).

    :param array:
    :param pixel_size:
    :param x_min:
    :param y_max:
    :param wkt:
    :param path:
    :return:
    """
    # FIXME not done yet. What to return? I think geopandas may be useful for all this polygon stuff.

    # Convert array to shapely polygons
    # Read into generator
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v)
            in enumerate(shapes(array))
    )
    geoms = list(results)

def utm_lookup(zone):
    """Returns a wkt string of a given UTM zone. Used as a bypass for older las file specifications that do not
    contain wkt strings.


    :param zone: The UTM zone (as a string)
        ex: "10N"
    """
    pcs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pcs.csv')
    in_ds = ogr.GetDriverByName('CSV').Open(pcs_path)
    layer = in_ds.GetLayer()
    layer.SetAttributeFilter("COORD_REF_SYS_NAME LIKE '%UTM zone {}%'".format(zone))
    wkt_string = []
    for feature in layer:
        code = feature.GetField("COORD_REF_SYS_CODE")
        name = feature.GetField("COORD_REF_SYS_NAME")
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(code))
        wkt_string.append(srs.ExportToWkt())
    return ''.join(wkt_string)
