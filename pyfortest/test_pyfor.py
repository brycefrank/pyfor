from pyfor import *
import unittest

# modeled heavily after laspytest
# https://github.com/laspy/laspy/blob/master/laspytest/test_laspy.py
import pandas as pd
import laspy
import os

test_las = os.path.abspath('data/test.las')

class CloudDataTestCase(unittest.TestCase):
    def setUp(self):
        self.test_points = {
            "x" : [0, 1],
            "y" : [0, 1],
            "z" : [0, 1],
            "intensity" : [0, 1],
            "classification" : [0, 1],
            "flag_byte" : [0, 1],
            "scan_angle_rank" : [0, 1],
            "user_data" : [0, 1],
            "pt_src_id" : [0, 1]
        }

        self.test_header = laspy.file.File("").header

        self.test_points = pd.DataFrame.from_dict(self.test_points)
        self.column = [0,1]
        self.test_cloud_data = cloud.CloudData(self.test_points, self.test_header)


    def test_init(self):
        self.assertEqual(type(self.test_cloud_data), cloud.CloudData)

    def test_data_length(self):
        self.assertEqual(len(self.test_cloud_data.points), 2)


    def test_write(self):
        self.test_cloud_data.write("data/temp_test_write.las")
        read = laspy.file.File('data/temp_test_write.las')
        self.assertEqual(type(read), laspy.file.File)
        read.close()

        os.remove('data/temp_test_write.las')

    def tearDown(self):
        self


class CloudTestCase(unittest.TestCase):

    def setUp(self):
        self.test_cloud = cloud.Cloud("data/test.las")

    def test_las_load(self):
        """Tests if a .las file succesfully loads when cloud.Cloud is called"""
        self.assertEqual(type(self.test_cloud), cloud.Cloud)

    def test_grid_creation(self):
        """Tests if the grid is successfully created."""
        # Does the call to grid return the proper type
        self.assertEqual(type(self.test_cloud.grid(1)), rasterizer.Grid)

    def test_filter_works(self):
        pass

    # TODO Come up with adequate tests for plotting methods

    def test_clip_square(self):
        mins, maxes = self.test_cloud.las.header.min, self.test_cloud.las.header.max
        clip_cloud = self.test_cloud.clip((mins[0], mins[0]+5, mins[1], mins[1]+5))
        self.assertEqual(type(clip_cloud), cloud.Cloud)
        self.assertLess(clip_cloud.las.count, self.test_cloud.las.count)
        pass

    def test_clip_polygon(self):
        pass

    def test_clip_circle(self):
        pass

class GridTestCase(unittest.TestCase):
    def setUp(self):
        self.test_grid = cloud.Cloud("data/test.las").grid(1)

    def test_empty_cells(self):
        empty = self.test_grid.empty_cells
        self.assertEqual(empty.shape, (167, 2))

    def tearDown(self):
        del self.test_grid.las.header



