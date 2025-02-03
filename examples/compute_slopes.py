from pathlib import Path

from bathyu import AlignedRasters
from bathyu.io.export import to_nc

# Find all geotiffs in this folder
files = sorted(
    Path(r"p:\11210344-scheldemonding\01_Data\02_Bodemdata\multibeam\NetCDF").glob(
        "*.tiff"
    )
)

# Organize them into an instance of AlignedRasters
rasters = AlignedRasters.from_files(files, "%Y%m%d", resolution=1, force_nodata=False)

# Compute all slopes of individual rasters
slope = rasters.slope()

# Export the slopes to NetCDF
to_nc(
    slope,
    r"p:\11211195-co2-noordzee\Data\westerscheldemonding_multibeam_2022-2024_slopes.nc",
    compress=True,
    compress_level=9,
)
