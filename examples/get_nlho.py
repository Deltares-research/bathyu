from pathlib import Path

import geopandas as gpd
import numpy as np
import rioxarray as rio
import xarray as xr

from bathyu.io.export import to_geotiff
from bathyu.nlho.nlho import combine_nlho_mosaics
from bathyu.rastercalc import cell_coverage


def temporal_coverage(dataset, since="1990-01-01", window=365):
    valid_times = (dataset.time - np.datetime64(since)) / np.timedelta64(1, "D") > 0
    data_sel = dataset.isel(time=valid_times)
    total_days = (
        np.datetime64("2025-01-01") - np.datetime64("1990-01-01")
    ) / np.timedelta64(1, "D")
    temporal_coverage = (
        (cell_coverage(data_sel.z, axis=0) * total_days) / total_days
    ) - 1
    temporal_coverage_da = xr.DataArray(
        temporal_coverage,
        coords=[data_sel.y, data_sel.x],
        dims=["y", "x"],
        name="temporal_coverage",
        attrs=data_sel.attrs,
    )
    return temporal_coverage_da


if __name__ == "__main__":
    folder = Path(r"p:\tgg-mariene-data\__UPDATES\NetCDF_test")
    files = list(folder.glob("*.nc"))

    combined_mosaics = combine_nlho_mosaics(files)

    datasets = [xr.open_dataset(file) for file in files]
    temp_cov = [temporal_coverage(dataset) for dataset in datasets]
    norm_cov = [dataset.coverage for dataset in datasets]

    combined = xr.combine_by_coords(norm_cov, combine_attrs="override")
    temporal_combined = xr.combine_by_coords(temp_cov, combine_attrs="override")

    to_geotiff(combined_mosaics.z, folder.joinpath("mosaic.tif"))
    to_geotiff(combined.coverage, folder.joinpath("coverage.tif"))
    to_geotiff(
        temporal_combined.temporal_coverage,
        folder.joinpath("temporal_coverage.tif"),
    )
