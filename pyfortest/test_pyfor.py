from pyfor import *
import unittest

# modeled heavily after laspytest
# https://github.com/laspy/laspy/blob/master/laspytest/test_laspy.py
import pandas as pd
import laspy
import os
import numpy as np
import geopandas as gpd
import plyfile
import matplotlib.pyplot as plt

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
test_las = os.path.join(data_dir, 'test.las')
test_ply = os.path.join(data_dir, 'test.ply')
test_laz = os.path.join(data_dir, 'test.laz')
test_shp = os.path.join(data_dir, 'clip.shp')
proj4str = "+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"

test_points = {
    "x": [0, 1, 2],
    "y": [0, 1, 2],
    "z": [0, 1, 2],
    "intensity": [0, 1, 2],
    "classification": [0, 1, 2],
    "flag_byte": [0, 1, 2],
    "scan_angle_rank": [0, 1, 2],
    "user_data": [0, 1, 2],
    "pt_src_id": [0, 1, 2],
    "return_num": [0, 1 ,2]
}

class DataTestCase:
    def test_init(self):
        self.assertEqual(type(self.test_data), self.test_type)

    def test_data_length(self):
        self.assertEqual(len(self.test_data.points), len(test_points['z']))


class PLYDataTestCase(unittest.TestCase, DataTestCase):
    def setUp(self):
        self.test_points = pd.DataFrame.from_dict(test_points).astype(np.float)
        self.test_header = 0
        self.test_data = cloud.PLYData(self.test_points, self.test_header)
        self.test_type = cloud.PLYData

    def test_write(self):
        self.test_data.write(os.path.join(data_dir, "temp_test_write.ply"))
        plyfile.PlyData.read(os.path.join(data_dir, 'temp_test_write.ply'))
        os.remove(os.path.join(data_dir, "temp_test_write.ply"))


class LASDataTestCase(unittest.TestCase, DataTestCase):
    def setUp(self):
        self.test_points = pd.DataFrame.from_dict(test_points)
        self.test_header = laspy.file.File(test_las).header
        self.column = [0,1]
        self.test_data = cloud.LASData(self.test_points, self.test_header)
        self.test_type = cloud.LASData

    def tearDown(self):
        self.test_data.header.reader.close()

    def test_write(self):
        self.test_data.write(os.path.join(data_dir, "temp_test_write.las"))
        read = laspy.file.File(os.path.join(data_dir, "temp_test_write.las"))
        self.assertEqual(type(read), laspy.file.File)
        read.close()
        os.remove(os.path.join(data_dir, "temp_test_write.las"))

class CloudTestCase:
    def test_not_supported(self):
        with self.assertRaises(ValueError):
            # Attempt to read a non-supported file type
            cloud.Cloud(os.path.join(data_dir, "clip.shp"))

    def test_cloud_summary(self):
        print(self.test_cloud)

    def test_grid_creation(self):
        """Tests if the grid is successfully created."""
        # Does the call to grid return the proper type
        self.assertEqual(type(self.test_cloud.grid(1)), rasterizer.Grid)

class LASCloudTestCase(unittest.TestCase, CloudTestCase):
    def setUp(self):
        self.test_cloud = cloud.Cloud(test_las)

    def tearDown(self):
        self.test_cloud.data.header.reader.close()

    def test_dimensions_loaded(self):
        local_cloud = cloud.Cloud(test_las)
        self.assertEqual(type(local_cloud), cloud.Cloud)

        if local_cloud.extension == '.las' or local_cloud.extension == '.laz':
            self.assertListEqual(list(local_cloud.data.points.columns.values),
                                 ["x", "y", "z", "intensity", "return_num", "classification",
                                  "flag_byte", "scan_angle_rank", "user_data", "pt_src_id"])

        local_cloud.data.header.reader.close()

    def test_filter_z(self):
        self.test_cloud.filter(40, 41, "z")
        self.assertEqual(self.test_cloud.data.count, 3639)
        self.assertLessEqual(self.test_cloud.data.max[2], [41])
        self.assertGreaterEqual(self.test_cloud.data.min[2], [40])

    def test_clip_polygon(self):
        poly = gpd.read_file(test_shp)['geometry'][0]
        self.test_cloud.clip(poly)

    def test_plot(self):
        self.test_cloud.plot()

    def test_plot3d(self):
        self.test_cloud.plot3d()
        self.test_cloud.plot3d(dim='user_data')

    def test_normalize(self):
        self.test_cloud.normalize(3)
        self.assertLess(self.test_cloud.data.max[2], 65)

    def test_normalize_classified(self):
        self.test_cloud.normalize(3, classified=True)
        self.assertLess(self.test_cloud.data.max[2], 65)

    def test_subtract(self):
        zf = ground_filter.Zhang2003(1)
        bem = zf.bem(self.test_cloud)
        bem.write('./temp_bem.tif')
        self.test_cloud.subtract('./temp_bem.tif')

        # In theory this should equal the z column from a regular normalization, do this on a separate cloud
        pc = cloud.Cloud(test_las)
        pc.normalize(1)
        normalize_z = pc.data.points['z']
        subtracted_z = self.test_cloud.data.points['z']

        # Allow a tolerance of 0.005 meters
        self.assertLess(np.mean(normalize_z.values) - np.mean(subtracted_z.values), 0.005)

        os.remove('./temp_bem.tif')

    def test_chm(self):
        self.test_cloud.chm(0.5, interp_method="nearest", pit_filter="median")

    def test_chm_without_interpolation_method(self):
        self.assertEqual(type(self.test_cloud.chm(0.5, interp_method=None)), rasterizer.Raster)

    def test_append(self):
        n_points = len(self.test_cloud.data.points)
        self.test_cloud.data._append(self.test_cloud.data)
        self.assertGreater(len(self.test_cloud.data.points), n_points)

    def test_write(self):
        self.test_cloud.write(os.path.join(data_dir, 'test_write.las'))
        os.remove(os.path.join(data_dir, 'test_write.las'))

# TODO broken on Travis
#class LAZCloudTestCase(LASCloudTestCase):
#    def setUp(self):
#        self.test_cloud = cloud.Cloud(test_laz)

class PLYCloudTestCase(unittest.TestCase, CloudTestCase):
    def setUp(self):
        self.test_cloud = cloud.Cloud(test_ply)

    def test_write(self):
        self.test_cloud.write(os.path.join(data_dir, 'test_write.ply'))
        os.remove(os.path.join(data_dir, 'test_write.ply'))

class GridTestCase(unittest.TestCase):
    def setUp(self):
        self.test_grid = cloud.Cloud(test_las).grid(1)

    def tearDown(self):
        self.test_grid.cloud.data.header.reader.close()

    def test_m(self):
        self.assertEqual(200, self.test_grid.m)

    def test_n(self):
        self.assertEqual(200, self.test_grid.n)

    def test_cloud(self):
        self.assertEqual(type(self.test_grid.cloud), cloud.Cloud)

    def test_empty_cells(self):
        empty = self.test_grid.empty_cells
        # Check that there are the correct number
        self.assertEqual(empty.shape, (291, 2))

    def test_raster(self):
        raster = self.test_grid.raster("max", "z")
        self.assertEqual(type(raster), rasterizer.Raster)

    def test_interpolate(self):
        self.test_grid.interpolate("max", "z")

    def test_update(self):
        """
        Change a few points from parent cloud, update and check if different
        :return:
        """
        pre = self.test_grid.m
        self.test_grid.cloud.data.points = self.test_grid.cloud.data.points.iloc[1:50]
        self.test_grid._update()
        post = self.test_grid.m


class GridMetricsTestCase(unittest.TestCase):
    def setUp(self):
        self.test_grid = cloud.Cloud(test_las).grid(20)

    def test_pct_above_mean(self):
        all_above_mean = metrics.pct_above_heightbreak(self.test_grid, r=0, heightbreak="mean")
        self.assertEqual(0, np.sum(all_above_mean.array > 1))
        r1_above_mean = metrics.pct_above_heightbreak(self.test_grid, r=1, heightbreak="mean")
        self.assertEqual(0, np.sum(r1_above_mean.array > 1))

    def test_pct_above_heightbreak(self):
        all_above_2 = metrics.pct_above_heightbreak(self.test_grid, r=0, heightbreak=2)
        self.assertEqual(0, np.sum(all_above_2.array > 1))
        r1_above_2 = metrics.pct_above_heightbreak(self.test_grid, r=1, heightbreak=2)
        self.assertEqual(0, np.sum(r1_above_2.array > 1))

    def test_return_num(self):
        rast = metrics.return_num(self.test_grid, 1)
        self.assertEqual(1220, rast.array[0,0])
        rast = metrics.return_num(self.test_grid, 100)
        self.assertTrue(np.isnan(rast.array[0,0]))

    def test_all_returns(self):
        rast = metrics.all_returns(self.test_grid)
        self.assertEqual(1531, rast.array[0,0])


    def test_total_returns(self):
        rast = metrics.total_returns(self.test_grid)
        self.assertEqual(1531, rast.array[0,0])

    def test_standard_metrics(self):
        metrics_dict = metrics.standard_metrics_grid(self.test_grid, 2)


class CloudMetrcsTestCase(unittest.TestCase):
    def setUp(self):
        self.test_cloud = cloud.Cloud(test_las)

    def test_standard_metrics(self):
        metrics.standard_metrics_cloud(self.test_cloud.data.points, 2)

class RasterTestCase(unittest.TestCase):
    def setUp(self):
        pc = cloud.Cloud(test_las)
        self.test_raster = pc.grid(1).raster("max", "z")
        self.test_raster.grid.cloud.crs = proj4str

    def tearDown(self):
        self.test_raster.grid.cloud.data.header.reader.close()

    def test_affine(self):
        affine = self.test_raster._affine
        self.assertEqual(affine[0], 1.0)
        self.assertEqual(affine[1], 0.0)
        self.assertEqual(affine[2], 405000.01000000001)
        self.assertEqual(affine[3], 0.0)
        self.assertEqual(affine[4], -1.0)
        self.assertEqual(affine[5], 3276499.9900000002)
        self.assertEqual(affine[6], 0)

    def test_array_oriented_correctly(self):
        """
        Tests if the index [0,0] refers to the top left corner of the image. That is, if I were to plot the raster
        using plt.imshow it would appear to the user as a correctly oriented image.
        """
        self.assertEqual(self.test_raster.array[0,0], 45.11)

    def test_plot(self):
        self.test_raster.plot()
        self.test_raster.plot(return_plot=True)

    def test_force_extent_contract(self):
        min_x, min_y = self.test_raster.grid.cloud.data.header.min[0:2]
        max_x, max_y = self.test_raster.grid.cloud.data.header.max[0:2]

        # Test buffering in 10 meters
        buffer = 10
        bbox = (min_x + buffer, max_x - buffer, min_y + buffer, max_y - buffer)

        self.test_raster.force_extent(bbox)
        self.assertEqual(self.test_raster.array.shape, (180, 180))
        self.assertEqual(self.test_raster._affine[2], min_x + buffer)
        self.assertEqual(self.test_raster._affine[5], max_y - buffer)

    def test_force_extent_expand(self):
        min_x, min_y = self.test_raster.grid.cloud.data.header.min[0:2]
        max_x, max_y = self.test_raster.grid.cloud.data.header.max[0:2]

        # Test buffering out 10 meters
        buffer = 10
        bbox = (min_x - buffer, max_x + buffer, min_y - buffer, max_y + buffer)

        self.test_raster.force_extent(bbox)
        self.assertEqual(self.test_raster.array.shape, (220, 220))
        self.assertEqual(self.test_raster._affine[2], min_x - buffer)
        self.assertEqual(self.test_raster._affine[5], max_y + buffer)

    def test_write_with_crs(self):
        self.test_raster.write("./temp_tif.tif")

    def test_write_without_crs(self):
        self.test_raster.crs = None
        self.test_raster.write("./temp_tif.tif")

class RetileTestCase(unittest.TestCase):
    def setUp(self):
        cdf = collection.from_dir(os.path.join(data_dir, 'mock_collection'))
        self.retiler = collection.Retiler(cdf)

    def test_retile_raster(self):
        self.retiler.retile_raster(10, 100)
        self.retiler.retile_raster(10, 100, 10)

    def test_retile_buffer(self):
        self.retiler.retile_buffer(10)


class GISExportTestCase(unittest.TestCase):
    def setUp(self):
        self.test_grid = cloud.Cloud(test_las).grid(1)
        self.test_raster = self.test_grid.raster("max", "z")

    def tearDown(self):
        self.test_raster.grid.cloud.data.header.reader.close()

    def test_project_indices(self):
        test_indices = np.array([[0,0], [1,1]])
        gisexport.project_indices(test_indices, self.test_raster)

    def test_pcs_exists(self):
        print(os.path.realpath(__file__))
        pcs_path = os.path.join('..', 'pyfor', 'pcs.csv', os.path.dirname(os.path.realpath(__file__)))
        self.assertTrue(os.path.exists(pcs_path))

    def test_array_to_raster_writes(self):
        test_grid = cloud.Cloud(test_las).grid(1)
        test_grid.cloud.crs = proj4str
        test_raster = test_grid.raster("max", "z")
        gisexport.array_to_raster(test_raster.array, test_raster._affine, proj4str, os.path.join(data_dir, "temp_raster_array.tif"))
        self.assertTrue(os.path.exists(os.path.join(data_dir, "temp_raster_array.tif")))
        os.remove(os.path.join(data_dir, "temp_raster_array.tif"))

    def test_raster_output_transform(self):
        """
        Tests if the written raster output was rotated and transformed correctly.
        :return:
        """
        pass

    def test_array_to_polygon(self):
        array = np.random.randint(1, 5, size=(99, 99)).astype(np.int32)
        gisexport.array_to_polygons(array, self.test_raster._affine)
        gisexport.array_to_polygons(array)

class VoxelGridTestCase(unittest.TestCase):
    def setUp(self):
        self.test_voxel_grid = voxelizer.VoxelGrid(cloud.Cloud(test_las), cell_size=2)

    def test_voxel_raster(self):
        self.test_voxel_grid.voxel_raster("count", "z")

class KrausPfeifer1998(unittest.TestCase):
    def setUp(self):
        self.test_cloud = cloud.Cloud(test_las)
        self.test_kp_filter = ground_filter.KrausPfeifer1998(3)

    def test_filter(self):
        self.test_kp_filter._filter(self.test_cloud.grid(self.test_kp_filter.cell_size))

class Zhang2003TestCase(unittest.TestCase):
    def setUp(self):
        self.test_cloud = cloud.Cloud(test_las)
        self.test_zhang_filter = ground_filter.Zhang2003(3)

    def test_filter(self):
        self.test_zhang_filter._filter(self.test_cloud.grid(self.test_zhang_filter.cell_size))

    def test_bem(self):
        self.test_zhang_filter.bem(self.test_cloud)
