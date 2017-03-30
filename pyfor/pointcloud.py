import laspy
import numpy as np
import pandas as pd


class CloudInfo:
    """Holds and manipulates the data from a .las point cloud."""
    def __init__(self, filename):

        # Read las file from filename.
        self.filename = filename
        self.las = laspy.file.File(filename, mode='r')

        # Gets some information useful for later processes.
        self.header = self.las.header
        self.mins = self.las.header.min
        self.maxes = self.las.header.max

        # Constructs dataframe of las information.
        self.new_stack = np.column_stack((self.las.x, self.las.y, self.las.z, self.las.intensity, self.las.return_num))
        self.dataframe = pd.DataFrame(self.new_stack, columns=['x','y','z','int','ret'])
        self.dataframe['classification'] = 1

        # Place holder variables.
        self.grid = None
        self.grid_x = None
        self.grid_y = None
        self.wkt = None
        self.dem_path = None

    def grid_constructor(self, step, output=False):
        """Sets self.grid to a list of tuples corresponding to the points of a 2D grid covering the extent
        of the input .las

        Keyword arguments:
        step -- Length of grid cell side in same units as .las
        output -- If true, generates an ESRI shapefile of the grid, not used for any processes in Pyfor.
        """

        print("Constructing grid.")

        min_xyz = self.mins
        max_xyz = self.maxes

        min_xyz = [int(coordinate) for coordinate in min_xyz] # Round all mins down.
        max_xyz = [int(np.ceil(coordinate)) for coordinate in max_xyz] # Round all maxes up.

        # Get each coordinate for grid boundaries list constructor.
        x_min = min_xyz[0]
        x_max = max_xyz[0]
        y_min = min_xyz[1]
        y_max = max_xyz[1]

        self.grid_x = [x for x in range(x_min, x_max+step, step)]
        self.grid_y = [y for y in range(y_min, y_max+step, step)]
        # TODO: This makes things to immutable.
        self.grid_step = step

        # TODO: Implement exporting this to a shapefile or similar.
        if output:
            pass

    def cell_sort(self):
        """Sorts cells into grid constructed by grid_constructor by appending cell_x & cell_y to self.dataframe."""

        print("Sorting cells into grid.")

        x_list = self.new_stack[:,0]
        y_list = self.new_stack[:,1]


        bins_x = np.digitize(x_list, self.grid_x)
        x = np.array(self.grid_x)
        x = x[bins_x-1]

        bins_y = np.digitize(y_list, self.grid_y)
        y = np.array(self.grid_y)
        y = y[bins_y-1]

        self.dataframe['cell_x'] = x
        self.dataframe['cell_y'] = y


    def ground_classify(self, method):
        """Classifies points in self.dataframe as 2 using a simple ground filter."""
        print("Classifying points as ground.")

        #Retrieve necessary dataframe fields.
        df = self.dataframe[['z', 'cell_x', 'cell_y']]
        if method == "simple":
            # Construct list of ID's to adjust
            #TODO: Could be sped up (see CHM function)
            grouped = df.groupby(['cell_x', 'cell_y'])
            ground_id = [df.idxmin()['z'] for key, df in grouped]

            #  Adjust to proper classification (2 used per las documentation).
            for coord_id in ground_id:
                self.dataframe.set_value(coord_id, 'classification', 2)

        # TODO: Implement other ground filter options.
        else:
            pass

    def point_cloud_to_dem(self, path):
        """Holds a variety of functions that create a GeoTIFF from a classified point cloud.

        Keyword arguments:
        path -- output path for GeoTIFF output.
        """
        from scipy.interpolate import griddata
        import gdal
        cloud = self.dataframe

        # TODO: Add a function that checks if any points are classified as 2.
        self.dem_path = path
        wkt = self.wkt

        def interpolate(cloud, resolution=1, int_method='cubic'):
            """Creates an interpolated 2d array from XYZ."""
            ground_df = cloud.loc[cloud['classification'] == 2]  # Retrieves ground points from data frame.
            ground_points = ground_df.as_matrix(['x', 'y','z'])  # Converts ground points to numpy array.
            # FIXME: Ground_points and nodes are verry similar.

            def extrap_corners():

                # Find the corners of the las extent.
                # TODO: Clean this up

                s = self.grid_step

                top_left = [self.mins[0]-s, self.maxes[1]+s]
                top_right = [self.maxes[0]+s, self.maxes[1]+s]
                bot_left = [self.mins[0]-s, self.mins[1]-s]
                bot_right = [self.maxes[0]+s, self.mins[1]-s]

                # Add other control points

                top = [[x, self.maxes[1]+s] for x in self.grid_x]
                bot = [[x, self.mins[1]-s] for x in self.grid_x]
                left = [[self.mins[0]-s, y] for y in self.grid_y]
                right = [[self.maxes[0]+s, y] for y in self.grid_y]

                control_coords = [top_left, top_right, bot_left, bot_right, *top, *bot,
                                *left, *right]
                # Get the XYZ information of the classified ground pixels.
                nodes = np.asarray(list(zip(ground_df.x, ground_df.y)))
                corner_ground = []
                for coord in control_coords:
                    dist_2 = np.sum((nodes - coord)**2, axis=1)
                    index = np.argmin(dist_2)
                    corner_ground.append([coord[0], coord[1], ground_points[index][2]])

                return corner_ground

            ground_points = np.vstack([ground_points, extrap_corners()])
            x_min = int(np.amin(ground_points, axis=0)[0])
            x_max = int(np.amax(ground_points, axis=0)[0])
            y_min = int(np.amin(ground_points, axis=0)[1])
            y_max = int(np.amax(ground_points, axis=0)[1])
            # TODO: This could be more elegant, do it properly.
            grid_x, grid_y = np.mgrid[x_min:x_max:resolution, y_min:y_max:resolution]
            return griddata(ground_points[:,:2],ground_points[:,2], (grid_x, grid_y), method=int_method)

        def cloud_to_tiff(cloud, wkt, path, int_method='cubic', resolution=1):
            from PyFor.pyfor import gisexport
            gisexport.array_to_raster(interpolate(cloud, resolution, int_method), resolution,self.mins[0], self.maxes[1], wkt, path)

        if wkt == None:
            print("This point cloud object does not have a wkt. Add one with the add_wkt function.")
        else:
            print("Converting ground points to GeoTIFF")
            cloud_to_tiff(cloud, wkt, path, resolution=1)

    def generate_BEM(self, step, tiff_path, method="simple"):
        self.grid_constructor(step)
        self.cell_sort()
        self.ground_classify(method="simple")
        self.point_cloud_to_dem(tiff_path)
        self.dem_path = tiff_path


    def rasterize(self, step, func, path):
        """Rasterizes a summary function over the grouped dataframe."""
        from PyFor.pyfor import gisexport
        self.grid_constructor(step, output=False)
        self.cell_sort()
        x_width = len(self.grid_x)-1
        y_width = len(self.grid_y)-1

        grouped  = self.dataframe.groupby(['cell_x', 'cell_y'])
        cell_values = grouped['z'].aggregate(func).values
        cell_values = np.reshape(cell_values, (x_width, y_width))

        gisexport.array_to_raster(cell_values, step, self.mins[0],
                                  self.maxes[1], self.wkt, path)

    def normalize(self, dem_path,  export=False, path=None):
        from PyFor.pyfor import normalize
        normalize.elev_points(dem_path, self)
        if export:
            normalize.df_to_las(self.dataframe, path, self.header)

class GriddedCloud(CloudInfo):
    """Creates a grid object useful for extracting information from CloudInfo"""
    def __init__(self, filename, step):
        super().__init__(filename)
        self.step = step
        self.grid_constructor(step)

