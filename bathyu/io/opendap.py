import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import rioxarray as rio
import xarray as xr
from bs4 import BeautifulSoup
from tqdm import tqdm

# Deltares NLHO OpenDAP settings
NLHO_TILE_RESOLUTION = 5000
NLHO_LOWER_X_LIM = 405000
NLHO_UPPER_X_LIM = 780000
NLHO_LOWER_Y_LIM = 5660000
NLHO_UPPER_Y_LIM = 6185000


def list_opendap_files(
    base_url: str,
) -> list[str]:
    """
    List files available at an OPeNDAP base URL.
    This function sends a GET request to the specified OPeNDAP base URL and parses the HTML response to extract
    the list of files available at that URL. It returns a list of file names.

    Parameters
    ----------
    base_url : str, optional
        The base URL of the OPeNDAP server to list files from. Default is
        'https://opendap.deltares.nl/thredds/dodsC/opendap/hydrografie/surveys_2025'.

    Returns
    -------
    list of str
        A list of file names available at the specified OPeNDAP base URL. If the request fails, an empty list is returned.

    Examples
    --------
    >>> list_opendap_files()
    ['file1.nc', 'file2.nc', 'file3.nc']
    >>> list_opendap_files("https://example.com/opendap")
    ['example1.nc', 'example2.nc']
    """
    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Failed to access {base_url}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    files = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and not href.startswith("?") and not href.startswith("/"):
            files.append(href.split("/")[-1])

    return files


def nlho_tiles_from_bbox(
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    base_url: str = r"https://opendap.deltares.nl/thredds/dodsC/opendap/hydrografie/surveys_2025",
    **kwargs,
) -> list[xr.Dataset]:
    """
    Retrieve NLHO tiles within a specified bounding box from the OpenDAP server.

    Parameters
    ----------
    xmin : int
        Minimum x-coordinate of the bounding box.
    ymin : int
        Minimum y-coordinate of the bounding box.
    xmax : int
        Maximum x-coordinate of the bounding box.
    ymax : int
        Maximum y-coordinate of the bounding box.
    base_url : str, optional
        Base URL for the OpenDAP server (default is "https://opendap.deltares.nl/thredds/dodsC/opendap/hydrografie/surveys_2025").
    **kwargs : dict, optional
        Additional keyword arguments to pass to `xr.open_dataset`.

    Returns
    -------
    list of xarray.Dataset
        List of xarray datasets for each tile within the bounding box.

    Notes
    -----
    The bounding box coordinates are adjusted and aligned to the tile resolution.
    Only tiles available on the server are retrieved.
    """
    # Adjust and align bbox values to tile resolution
    xmin = max(xmin, NLHO_LOWER_X_LIM) // NLHO_TILE_RESOLUTION * NLHO_TILE_RESOLUTION
    ymin = max(ymin, NLHO_LOWER_Y_LIM) // NLHO_TILE_RESOLUTION * NLHO_TILE_RESOLUTION
    xmax = min(xmax, NLHO_UPPER_X_LIM) // NLHO_TILE_RESOLUTION * NLHO_TILE_RESOLUTION
    ymax = min(ymax, NLHO_UPPER_Y_LIM) // NLHO_TILE_RESOLUTION * NLHO_TILE_RESOLUTION

    # Get tile indices
    x_tile_indices = np.arange(xmin, xmax+1, NLHO_TILE_RESOLUTION)
    y_tile_indices = np.arange(ymin, ymax+1, NLHO_TILE_RESOLUTION)

    # List available files on the server
    catalog_url = base_url.replace("dodsC", "catalog") + "/catalog.html"
    available_files = [
        base_url + "/" + file for file in list_opendap_files(catalog_url)
    ]

    # Generate URLs for each tile
    tile_urls = [
        f"{base_url}/x{x_idx}y{y_idx}.nc"
        for x_idx, y_idx in product(x_tile_indices, y_tile_indices)
    ]

    tile_data = [
        xr.open_dataset(tile_url, **kwargs)
        for tile_url in tqdm(tile_urls)
        if tile_url in available_files
    ]
    return tile_data


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


if __name__ == "__main__":
    ds = nlho_tiles_from_bbox(580000, 5890000, 690000, 5960000)
    print(ds)
