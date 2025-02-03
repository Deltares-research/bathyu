from pathlib import Path

from bathyu import AlignedRasters
from bathyu.io.export import to_geotiff

# Find all vaklodingen in the KPP GIS data folder
files = sorted(
    [
        p
        for p in Path(
            r"p:\kpp-benokust-gis-data\bathymetrie\Vaklodingen\Mosaicen"
        ).glob("vakl_*")
        if p.is_dir()
    ]
)

# Organize them into an instance of AlignedRasters
rasters = AlignedRasters.from_files(
    files,
    "%Y",
    resolution=20,
    bounds="user",
    bbox=[110000, 120000, 550000, 560000],
)

# Compute most recent measurement and corresponding index (to see which measuerment
# is most recent at any location)
most_recent_mosaic = rasters.most_recent()
most_recent_idx = rasters.most_recent_index()

# Export results to geotiffs
to_geotiff(
    most_recent_mosaic,
    r"p:\kpp-benokust-gis-data\bathymetrie\Vaklodingen\Mosaicen\_Totaal_Mosaic\voorbeeld_most_recent.tiff",
    compress=True,
)
to_geotiff(
    most_recent_mosaic,
    r"p:\kpp-benokust-gis-data\bathymetrie\Vaklodingen\Mosaicen\_Totaal_Mosaic\voorbeeld_most_recent_index.tiff",
    compress=True,
)
