import importlib.machinery
import numpy as np
pyfor = importlib.machinery.SourceFileLoader('pyfor','/home/bryce/Programming/PyFor/pyfor/__init__.py').load_module()

pc = pyfor.cloud.Cloud("/home/bryce/Programming/PyFor/pyfortest/data/test.las")

pc.normalize(0.5)

pc.las.header.min