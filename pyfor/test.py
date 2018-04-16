import importlib.machinery
import numpy as np
pyfor = importlib.machinery.SourceFileLoader('pyfor','/home/bryce/Programming/PyFor/pyfor/__init__.py').load_module()

pc = pyfor.cloud.Cloud("/home/bryce/Desktop/pyfor_test_data/cercedilla_tiles/000001.S.CLAS.las")


