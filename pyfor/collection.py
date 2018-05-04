import pathlib
import laspy

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

    @property
    def _las_objects(self):
        """
        Returns a list of laspy las objects.
        """
        return [laspy.file.File(las_file) for las_file in self.las_paths]

    @property
    def las_headers(self):
        """
        Returns a list of las headers.
        """
        return [las_obj.header for las_obj in self._las_objects]