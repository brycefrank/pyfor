import os
import laspy
import pandas as pd
from joblib import Parallel, delayed
from pyfor import cloud


class CloudDataFrame(pd.DataFrame):
    """
    Just an idea class for now.
    """
    def __init__(self, *args, **kwargs):
        super(CloudDataFrame, self).__init__(*args, **kwargs)
        self.n_threads = 1


    @classmethod
    def from_dir(cls, las_dir, n_threads = 1):
        las_path_init = [[os.path.join(root, file) for file in files] for root, dirs, files in os.walk(las_dir)][0]
        cdf = CloudDataFrame({'las_paths': las_path_init})
        cdf.n_threads = n_threads
        return(cdf)

    def par_apply(self, func, column):
        """
        Apply a function to each las path. Allows for parallelization using the n_jobs argument. This is achieved \
        via joblib Parallel and delayed.

        :param func: The user defined function, must accept a single argument, the path of the las file.
        :param n_jobs: The nlumber of threads to spawn, default of 1.
        """
        output = Parallel(n_jobs=self.n_threads)(delayed(func)(plot_path) for plot_path in self[column])
        return output

    def _get_bounding_box(self, las_path):
        """
        Vectorized function to get a bounding box from an individual las path.
        :param las_path:
        :return:
        """
        # TODO Could be quicker to do a strictly laspy implementation here but that would also prevent arbitirary
        # segmentation of point clouds
        pc = cloud.Cloud(las_path)
        min_x, max_x = pc.las.min[0], pc.las.max[0]
        min_y, max_y = pc.las.min[1], pc.las.max[1]
        return((min_x, max_x, min_y, max_y))

    def _get_bounding_boxes(self):
        """
        Retrieves a bounding box for each path in las path.
        :return:
        """
        return self.par_apply(self._get_bounding_box, column="las_paths")

    def _build_polygons(self):
        """Builds the shapely polygons of the bounding boxes and adds them to self.data"""
        from shapely.geometry import Polygon
        bboxes = self._get_bounding_boxes()
        self["bounding_boxes"] = [Polygon(((bbox[0], bbox[2]), (bbox[1], bbox[2]),
                                                (bbox[1], bbox[3]), (bbox[0], bbox[3]))) for bbox in bboxes]

    def _buffer_polygon(self, polygon, buffer_distance):
        """Have to write this to get around lambda joblib error...."""
        return polygon.buffer(buffer_distance)

    @property
    def _constructor(self):
        return CloudDataFrame

def from_dir(las_dir, n_threads = 1):
    """
    Constructs a CloudDataFrame from a directory of las files. Just a wrapper for awkward syntax.
    :param las_dir:
    :return:
    """

    return CloudDataFrame.from_dir(las_dir, n_threads = n_threads)
