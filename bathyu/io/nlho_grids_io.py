import re
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as rio
import xarray as xr


def find_nlho_surveys_in_bbox(
    nlho_folder: Path | str, xmin: int, ymin: int, xmax: int, ymax: int
):
    files = list(Path(nlho_folder).glob("*.asc"))

    # Filter files based on given bounds
    selected_files = [
        f
        for f in files
        if (int(re.search(r"x(\d+)", f.stem).group(1)) >= xmin)
        & (int(re.search(r"x(\d+)", f.stem).group(1)) < xmax)
        & (int(re.search(r"y(\d+)", f.stem).group(1)) >= ymin)
        & (int(re.search(r"y(\d+)", f.stem).group(1)) < ymax)
    ]

    # Get unique tiles
    tile_search_pattern = re.compile(r"^(.*?_.*?_)")
    tiles = np.unique(
        [tile_search_pattern.search(f.stem).group(1) for f in selected_files]
    )

    included_surveys = []
    for tile in tiles:
        tile_files = [
            f
            for f in selected_files
            if tile_search_pattern.search(f.stem).group(1) == tile
        ]
        included_surveys += [f.stem.split("_")[2] for f in tile_files]

    return list(set(included_surveys))


def reconstruct_nlho_survey(
    nlho_folder,
    survey_id: str,
    within_bbox: bool = False,
    xmin: int = 0,
    ymin: int = 0,
    xmax: int = 0,
    ymax: int = 0,
    add_timestamp: bool = False,
):
    files = list(Path(nlho_folder).glob("*.asc"))

    if within_bbox:
        # Filter files based on given bounds
        files = [
            f
            for f in files
            if (int(re.search(r"x(\d+)", f.stem).group(1)) >= xmin)
            & (int(re.search(r"x(\d+)", f.stem).group(1)) < xmax)
            & (int(re.search(r"y(\d+)", f.stem).group(1)) >= ymin)
            & (int(re.search(r"y(\d+)", f.stem).group(1)) < ymax)
        ]

    survey_tile_data = [
        rio.open_rasterio(f)
        .squeeze()
        .drop_vars(["band", "spatial_ref"])
        .rio.write_crs(32631)
        for f in files
        if survey_id in f.stem
    ]
    data_combined = xr.combine_by_coords(survey_tile_data)
    data_combined.rio.write_nodata(
        data_combined.rio.nodata, encoded=False, inplace=True
    )
    data_combined = data_combined.where(
        data_combined != data_combined.rio.nodata, np.nan
    )
    data_combined.attrs["_FillValue"] = np.nan

    return data_combined
