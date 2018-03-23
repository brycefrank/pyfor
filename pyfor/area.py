import pointcloud2

class Collection:
    """Constructs a collection for multiple tiled project areas."""
    def __init__(self, las_dir):
        self.cloud_list = [CloudInfo(cloud) for cloud in las_dir]

    def get_neighbors(self):
        # Builds an array of neighbors for every cloud in cloud_list
        pass
