from bathyu.io.opendap import nlho_tiles_from_bbox

# Retrieving NLHO tiles from the Deltares OpenDAP server is as simple as providing the
# bounding box of the area of interest. The function returns a list of xarray datasets
# for each tile within the bounding box.
tiles = nlho_tiles_from_bbox(555000, 5765000, 575000, 5790000)
print(tiles[0])
