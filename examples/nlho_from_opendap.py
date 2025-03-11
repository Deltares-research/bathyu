from bathyu.io.opendap import nlho_tiles_from_bbox
from bathyu.rastercalc import most_recent

# Retrieving NLHO tiles from the Deltares OpenDAP server is as simple as providing the
# bounding box of the area of interest. The function returns a list of xarray datasets
# for each tile within the bounding box.
tiles = nlho_tiles_from_bbox(555000, 5765000, 575000, 5790000)

# For each tile, create a mosaic of the most recent non-NaN values along the time axis.
# The most_recent function, when given an xarray.Dataset, requires the dimension along
# which to perform the operation ("time") and the data variable to use ("z").
mosaics = [most_recent(tile, "time", "z") for tile in tiles]

# Plot one of the mosaics for demonstration:
mosaics[0].plot.imshow()


