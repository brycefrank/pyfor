import os, sys
import ogr




import pointcloud
import numpy as np
cloud1 = pointcloud.CloudInfo(r"C:\lidarproject\WA_Olympic_Peninsula_2013_000191\WA_Olympic_Peninsula_2013_000191.las")
cloud1.grid_constructor(10)

xs = cloud1.grid_x
ys = cloud1.grid_y


mesh = np.meshgrid(xs,ys)



def vertices(origin_x, origin_y):
    """Returns a list of the vertices of a grid cell at origin_x and origin_y"""
    try:
        top_left = (float(mesh[0][origin_y][origin_x]) , float(mesh[1][origin_y][origin_x]))
        top_right = (float(mesh[0][origin_y][origin_x+1]) , float(mesh[1][origin_y][origin_x+1]))
        bottom_left = (float(mesh[0][origin_y+1][origin_x]) , float(mesh[1][origin_y+1][origin_x]))
        bottom_right = (float(mesh[0][origin_y+1][origin_x+1]) , float(mesh[1][origin_y+1][origin_x+1]))
        return [top_left, top_right, bottom_right, bottom_left]
    except IndexError:
        pass


def vert_list(xs, ys):
    """Compiles the vertices of a grid into a list of square vertices for each grid cell."""
    a=0
    b=0
    verts = []
    for row in ys:
        for col in xs:
            if vertices(a,b) != None:
                verts.append(vertices(a, b))
            else:
                pass
            a+=1
        a=0
        b+=1
    return verts

def export_grid(outpath="grid.shp"):
    # TODO: Messy and ad-hoc, need to define projection after output somehow. Bugs with extent.
    # try geometry.AssignSpatialReference

    ## CREATE THE POLYGONS!##

    if os.path.exists(outpath):
        os.remove(outpath)

    # create output file
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    outDataSource = outDriver.CreateDataSource(outpath)
    outLayer = outDataSource.CreateLayer(outpath, geom_type=ogr.wkbPolygon)
    featureDefn = outLayer.GetLayerDefn()


    for vert_set in vert_list(xs, ys):
        ring = ogr.Geometry(ogr.wkbLinearRing)
        first_point = vert_set[0]
        for point in vert_set:
            ring.AddPoint(point[0],point[1])
        ring.AddPoint(first_point[0], first_point[1])
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(poly)

        outLayer.CreateFeature(outFeature)
        outFeature.Destroy()

def get_geom(path):
    """Reads a shapefile and returns a list of Wkt geometries."""
    geoms = []
    shapefile = path
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shapefile, 0)
    layer = dataSource.GetLayer()

    for feature in layer:
        geom = feature.GetGeometryRef()
        geoms.append(geom.ExportToWkt())
    return geoms




def intersect_plots():
    """Exports a list of JSON coordinates of cell vertices near the plot location."""
    testplot = get_geom(r"C:\pyfor\gis\fieldplotsbuff.shp")[0]
    testplot = ogr.CreateGeometryFromWkt(testplot)
    grid = get_geom(r"C:\pyfor\gis\grid.shp")
    plot_cells = []
    for cell in grid:
        cell = ogr.CreateGeometryFromWkt(cell)
        if testplot.Overlaps(cell) == True:
            json_cell = cell.ExportToJson()
            plot_cells.append(json_cell)
    return plot_cells
# TODO: could be used for cell las writing.

def extract_points(geoms):
    import json
    dicts = []
    for json_geom in geoms:
        ring_dict = json.loads(json_geom)
        dicts.append(ring_dict)
    master_coords = []
    for dict in dicts:
        master_coords += [coord_pair for coord_pair in dict['coordinates'][0]]


    import itertools
    master_coords.sort()
    unique_points=list(master_coords for master_coords,_ in itertools.groupby(master_coords))
    unique_x = [point[0] for point in unique_points]
    unique_y = [point[1] for point in unique_points]

    return [unique_x, unique_y]

cloud1.cell_sort()
dataframe = cloud1.dataframe

def groupby():
    pass



def df_sort():
    """Takes all of the unique vertices and makes a new dataframe"""
    # FIXME: Very slow. Consider using pandas join.
    import pandas as pd
    df = pd.DataFrame
    unique_x = extract_points(intersect_plots())[0]
    unique_y = extract_points(intersect_plots())[1]

    df = (dataframe.loc[(cloud1.dataframe["cell_x"].isin(unique_x)) & (cloud1.dataframe["cell_y"].isin(unique_y))])

    return df


def norm_las(cloud, out_path):
    """Exports normalized points to new las."""
    # FIXME: Not working.
    import laspy
    header = laspy.file.File(r"C:\lidarproject\WA_Olympic_Peninsula_2013_000191\WA_Olympic_Peninsula_2013_000191.las").header

    outfile = laspy.file.File(out_path, mode="w", header = header)
    outfile.x = cloud['x']
    outfile.y = cloud['y']
    outfile.z = cloud['z']



def extract_circle():
    """converts dataframe points to something"""
    plot = get_geom(r"C:\pyfor\gis\fieldplotsbuff.shp")[0]
    plot = ogr.CreateGeometryFromWkt(plot)

    plot_df = df_sort()

    point_array = np.column_stack([plot_df.index.values, plot_df.x.values, plot_df.y.values])

    plot_points = []
    for point in point_array:
        thing = ogr.Geometry(ogr.wkbPoint)
        thing.AddPoint(point[1],point[2])
        if thing.Within(plot):
            plot_points.append(point[0])

    new_df = plot_df.loc[plot_points,:]
    return new_df

# plot = get_geom(r"C:\pyfor\gis\fieldplotsbuff.shp")[0]
# plot = ogr.CreateGeometryFromWkt(plot)
# print(type(plot))
#
# # intersection = plot.Intersection(xy_to_point_geom())
# #
# # ogr.Geometry(ogr.wkb)
# #
# #
# # outDriver = ogr.GetDriverByName('ESRI Shapefile')
# #
# #
# # outpath="points.shp"
# #
# # if os.path.exists(outpath):
# #     os.remove(outpath)
#
# # create output file
# outDriver = ogr.GetDriverByName('ESRI Shapefile')
# outDataSource = outDriver.CreateDataSource(outpath)
# outLayer = outDataSource.CreateLayer(outpath, geom_type=ogr.wkbMultiPoint)
# featureDefn = outLayer.GetLayerDefn()
# outFeature = ogr.Feature(featureDefn)
# outFeature.SetGeometry(intersection)
# outLayer.CreateFeature(outFeature)
# outFeature.Destroy()


