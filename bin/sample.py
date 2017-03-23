from PyFor import pyfor

# TODO: Replace this testlas with a larger extent (300x300m?), low density las file.

# Read a las into pyfor's CloudInfo container.
cloud = pyfor.pointcloud.CloudInfo(r"..\pyfor\pyfortest\data\testlas.las")

# Sets the ground filter grid width to 10 and outputs the BEM as "bem.tiff" using the "simple" ground filter:
cloud.generate_BEM(10, "bem.tiff", method="simple")

# Normalizes the cloud object using the previously generated bem.tiff and outputs the new cloud as "normalized.las":
cloud.normalize(export=True, path="normalized.las")

#Create a sampler object
sample = pyfor.sampler.Sampler(cloud)

# Add plot shapefile to sampler object.
sample.plot_shp = "fieldplots.shp"

# Clip and export individual las files.
sample.clip_plots()