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

    @classmethod
    def from_dir(cls, las_dir):
        las_path_init = [[os.path.join(root, file) for file in files] for root, dirs, files in os.walk(las_dir)][0]
        return CloudDataFrame({'las_paths': las_path_init})

    @property
    def _constructor(self):
        return CloudDataFrame

class Collection:
    """
    Holds a collection of cloud objects for batch processing. This is preferred if you are trying to process many \
    point clouds. For example it differs from using Cloud objects in a for loop because it delays loading clouds into
    memory until necessary.

    :param las_dir: The directory of las files to reference.
    """
    def __init__(self, las_dir, n_threads = 1):
        self.las_dir = las_dir
        self.las_paths = [filepath.absolute() for filepath in pathlib.Path(self.las_dir).glob('**/*')]
        self.data = CloudDataFrame({'las_paths': self.las_paths})
        self.n_threads = 1

    def apply(self, func):
        """
        Apply a function to each cloud object. Allows for parallelization using the n_jobs argument. This is achieved \
        via joblib Parallel and delayed.
        
        :param func: The user defined function, must accept a single argument, the path of the las file.
        :param n_jobs: The number of threads to spawn, default of 1.
        """
        output = Parallel(n_jobs=self.n_threads)(delayed(func)(plot_path) for plot_path in self.data['las_paths'])
        return output

    def _get_bounding_box(self, las_path):
        # TODO Could be quicker to do a strictly laspy implementation here but that would also prevent arbitirary
        # segmentation of point clouds
        pc = cloud.Cloud(las_path)
        min_x, max_x = pc.las.min[0], pc.las.max[0]
        min_y, max_y = pc.las.min[1], pc.las.max[1]
        return((min_x, max_x, min_y, max_y))

    def _get_bounding_boxes(self):
        return self.apply(self._get_bounding_box)

    def _build_polygons(self):
        """Builds the shapely polygons of the bounding boxes and adds them to self.data"""
        from shapely.geometry import Polygon
        bboxes = self._get_bounding_boxes()
        self.data["bounding_boxes"] = [Polygon(((bbox[0], bbox[2]), (bbox[1], bbox[2]),
                                                (bbox[1], bbox[3]), (bbox[0], bbox[3]))) for bbox in bboxes]


