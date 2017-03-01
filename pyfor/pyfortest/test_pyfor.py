from PyFor.pyfor import *

import unittest

def fun(x):
    return x+1

class PointCloudTest(unittest.TestCase):


    def setUp(self):

        #TODO: Make relative path.
        cloud = pointcloud.CloudInfo(r".\data\testlas.las")

        self.maxes = cloud.maxes
        self.mins = cloud.mins

    def test_mins(self):
        """Fetch and test minimums."""
        mins = [405400.0, 5280100.0, 206.49]
        self.assertListEqual(self.mins, mins)

    def test_maxes(self):
        """Fetch and test minimums."""
        maxes = [405424.99, 5280124.99, 242.58]
        self.assertListEqual(self.maxes, maxes)

    def test_early_dataframe(self):
        #TODO: Not sure how to best handle this
        pass