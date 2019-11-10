from pyfor import *
import unittest
import os

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
proj4str = "+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"


def test_buffered_func(pc, tile):
    pass


def test_byfile_func(las_path):
    pass

def test_func_arg(pc, tile, args):
    return(args["test_arg"])


def make_test_collection():
    """
    Splits the testing tile into 4 tiles to use for testing
    :return:
    """

    pc = cloud.Cloud(os.path.join(data_dir, "test.las"))

    # Sample to only 1000 points for speed
    pc.data.points = pc.data.points.sample(1000, random_state=12)

    tr = pc.data.points[
        (pc.data.points["x"] > 405100) & (pc.data.points["y"] > 3276400)
    ]
    tl = pc.data.points[
        (pc.data.points["x"] < 405100) & (pc.data.points["y"] > 3276400)
    ]
    br = pc.data.points[
        (pc.data.points["x"] > 405100) & (pc.data.points["y"] < 3276400)
    ]
    bl = pc.data.points[
        (pc.data.points["x"] < 405100) & (pc.data.points["y"] < 3276400)
    ]

    all = [tr, tl, br, bl]

    for i, points in enumerate(all):
        out = cloud.LASData(points, pc.data.header)
        out.write(os.path.join(data_dir, "mock_collection", "{}.las".format(i)))

    pc.data.header.reader.close()


class CollectionTestCase(unittest.TestCase):
    def setUp(self):
        self.test_col = collection.from_dir(os.path.join(data_dir, "mock_collection"))
        self.test_col_path = os.path.join(data_dir, "mock_collection")

    def test_create_index(self):
        self.test_col.create_index()
        lax_paths = [
            os.path.join(self.test_col_path, lax_file)
            for lax_file in os.listdir(self.test_col_path)
            if lax_file.endswith(".lax")
        ]
        self.assertEqual(len(lax_paths), 4)

    def test_retile_raster(self):
        self.test_col.retile_raster(10, 50, buffer=10)
        self.assertEqual(len(self.test_col.tiles), 16)
        self.test_col.reset_tiles()

    def test_par_apply_buff_index(self):
        self.test_col.create_index()
        self.test_col.retile_raster(10, 50, buffer=10)
        self.test_col.par_apply(test_buffered_func, indexed=True)

    def test_par_apply_buff_noindex(self):
        self.test_col.par_apply(test_buffered_func, indexed=False)

    def test_par_apply_by_file(self):
        self.test_col.par_apply(test_byfile_func, by_file=True)

    def test_par_apply_arg(self):
        self.test_col.par_apply(test_func_arg, indexed=False, args={"test_arg": 3})

    def tearDown(self):
        # Delete any .lax files
        lax_paths = [
            os.path.join(self.test_col_path, lax_file)
            for lax_file in os.listdir(self.test_col_path)
            if lax_file.endswith(".lax")
        ]
        for lax_path in lax_paths:
            os.remove(lax_path)
