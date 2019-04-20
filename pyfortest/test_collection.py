from pyfor import *
import unittest
import os

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
proj4str = "+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"

def test_buffered_func(pc, tile):
    print(pc, tile)

def test_byfile_func(las_path):
    print(cloud.Cloud(las_path))


def make_test_collection():
    """
    Splits the testing tile into 4 tiles to use for testing
    :return:
    """

    pc = cloud.Cloud(os.path.join(data_dir, 'test.las'))

    # Sample to only 1000 points for speed
    pc.data.points = pc.data.points.sample(1000, random_state=12)

    tr = pc.data.points[(pc.data.points['x'] > 405100) & (pc.data.points['y'] > 3276400)]
    tl = pc.data.points[(pc.data.points['x'] < 405100) & (pc.data.points['y'] > 3276400)]
    br = pc.data.points[(pc.data.points['x'] > 405100) & (pc.data.points['y'] < 3276400)]
    bl = pc.data.points[(pc.data.points['x'] < 405100) & (pc.data.points['y'] < 3276400)]

    all = [tr, tl, br, bl]

    for i, points in enumerate(all):
        out = cloud.LASData(points, pc.data.header)
        out.write(os.path.join(data_dir, 'mock_collection', '{}.las'.format(i)))


class CollectionTestCase(unittest.TestCase):
    def setUp(self):
        make_test_collection()
        self.test_col = collection.from_dir(os.path.join(data_dir, 'mock_collection'))

    def test_create_index(self):
        self.test_col.create_index()

    def test_retile_raster(self):
        self.test_col.retile_raster(10, 50, buffer=10)
        self.test_col.reset_tiles()

    def test_par_apply_buff_index(self):
        # Buffered with index
        self.test_col.retile_raster(10, 50, buffer=10)
        self.test_col.par_apply(test_buffered_func, indexed=True)

    def test_par_apply_buf_noindex(self):
        # Buffered without index
        self.test_col.par_apply(test_buffered_func, indexed=False)

    def test_par_apply_by_file(self):
        # By file
        self.test_col.par_apply(test_byfile_func, by_file=True)

