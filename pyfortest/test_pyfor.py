from pyfor import *
import unittest

# modeled heavily after laspytest
# https://github.com/laspy/laspy/blob/master/laspytest/test_laspy.py

class CloudTestCase(unittest.TestCase):

    def setUp(self):
        self.test_cloud = cloud.Cloud("data/test.las")
        self.test_grid = self.test_cloud.grid(1)

    def test_las_load(self):
        """Tests if a .las file succesfully loads when cloud.Cloud is called"""
        self.assertEqual(type(self.test_cloud), cloud.Cloud)
    def test_grid_creation(self):
        """Tests if the grid is successfully created."""
        # Does the call to grid return the proper type
        self.assertEqual(type(self.test_grid), rasterizer.Grid)
        # Are there bins?
        self.assertGreater(len(self.test_grid.data["bins_x"]), 0)
        self.assertGreater(len(self.test_grid.data["bins_y"]), 0)

