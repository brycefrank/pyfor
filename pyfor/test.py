import importlib.machinery
import numpy as np
pyfor = importlib.machinery.SourceFileLoader('pyfor','/home/bryce/Programming/PyFor/pyfor/__init__.py').load_module()

pc = pyfor.cloud.Cloud("/home/bryce/Programming/PyFor/pyfortest/data/test.las")


pc_grid = pc.grid(0.5)
pc.wkt = pyfor.gisexport.utm_lookup("10N")
chm = pc_grid.interpolate("max", "z")
chm = pyfor.rasterizer.Raster(chm, pc_grid)
tops = chm.watershed_seg()