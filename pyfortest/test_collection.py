from pyfor import *
import unittest
import os

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


def buffered_func(pc, tile):
    pass


def byfile_func(las_path):
    pass


def func_arg(pc, tile, args):
    return(args["test_arg"])


class CollectionTestCase(unittest.TestCase):
    def setUp(self):
        self.test_col = collection.from_dir(os.path.join(data_dir, "mock_collection"))

    def test_retile_raster(self):
        self.test_col.retile_raster(10, 50, buffer=10)
        self.assertEqual(len(self.test_col.tiles), 16)
        self.test_col.reset_tiles()

    def test_par_apply_buffered(self):
        self.test_col.retile_raster(10, 50, buffer=10)
        self.test_col.par_apply(buffered_func)

    def test_par_apply_by_file(self):
        self.test_col.par_apply(byfile_func, by_file=True)

    def test_par_apply_arg(self):
        self.test_col.par_apply(func_arg, args={"test_arg": 3})

    def test_datetimes_retrieval(self):
        timestamp = self.test_col['datetime'].iloc[0]
        timestamp = timestamp.to_pydatetime()
        self.assertEqual(timestamp.year, 2017)
