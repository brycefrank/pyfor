import ogr
import os

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


def export_coords_to_shp(coordlist, path):
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

# outDriver = ogr.GetDriverByName('ESRI Shapefile')
# outDataSource = outDriver.CreateDataSource(path)
# outLayer = outDataSource.CreateLayer(path, geom_type=ogr.wkbPolygon)
# featureDefn = outLayer.GetLayerDefn()
#
# for vert_set in vert_list(xs, ys):
#     ring = ogr.Geometry(ogr.wkbLinearRing)
#     first_point = vert_set[0]
#     for point in vert_set:
#         ring.AddPoint(point[0], point[1])
#     ring.AddPoint(first_point[0], first_point[1])
#     poly = ogr.Geometry(ogr.wkbPolygon)
#     poly.AddGeometry(ring)
#     outFeature = ogr.Feature(featureDefn)
#     outFeature.SetGeometry(poly)
#
#     outLayer.CreateFeature(outFeature)
#     outFeature.Destroy()
