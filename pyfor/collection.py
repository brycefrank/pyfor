import pathlib
import laspy
import pandas as pd
from joblib import Parallel, delayed

class Collection:
    """
    Holds a collection of cloud objects for batch processing. This is preferred if you are trying to process many \
    point clouds. For example it differs from using Cloud objects in a for loop because it delays loading clouds into
    memory until necessary.

    :param las_dir: The directory of las files to reference.
    """
    def __init__(self, las_dir):
        self.las_dir = las_dir
        self.las_paths = [filepath.absolute() for filepath in pathlib.Path(self.las_dir).glob('**/*')]
        self.data = pd.DataFrame({'las_paths': self.las_paths})

    def apply(self, func, n_jobs=1):
        """
        Apply a function to each cloud object. Allows for parallelization using the n_jobs argument. This is achieved \
        via joblib Parallel and delayed.
        
        :param func: The user defined function, must accept a single argument, the path of the las file.
        :param n_jobs: The number of threads to spawn, default of 1.
        """
        output = Parallel(n_jobs=n_jobs)(delayed(func)(plot_path) for plot_path in self.data['las_paths'])
        return output

    def plot(self):
        """
        
        :return: 
        """

