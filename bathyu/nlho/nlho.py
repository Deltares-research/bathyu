import getpass
import platform
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rioxarray as rio
import xarray as xr


def get_current_datetime(format: str = "%Y-%m-%dT%H:%MZ") -> str:
    """
    Get the current date and time formatted as a string.

    Parameters
    ----------
    format : str, optional
        The format in which to return the date and time. Default is "%Y-%m-%dT%H:%MZ".

    Returns
    -------
    str
        The current date and time formatted as a string.
    """
    return datetime.now().strftime(format)


def get_current_username() -> str:
    """
    Get the current system username.

    Returns
    -------
    str
        The username of the current user logged into the system.
    """
    return getpass.getuser()


def get_computer_name() -> str:
    """
    Get the name of the computer.

    Returns
    -------
    str
        The name of the computer as a string.
    """
    return platform.node()


def get_current_package_name() -> str:
    """
    Get the current package name from the file path.

    This function constructs the package name by joining the last four components
    of the file path, replacing backslashes with forward slashes.

    Returns
    -------
    str
        The package name derived from the file path.
    """
    return "/".join(str(Path(__file__)).split("\\")[-4:])


def parse_survey_metadata(metadata_excel: Path | str) -> pd.DataFrame:
    """
    Parses survey metadata from an Excel file.

    Parameters
    ----------
    metadata_excel : Path or str
        The path to the Excel file containing the survey metadata.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the concatenated metadata from all relevant sheets,
        with an additional 'survey_index' column set as the index.

    Notes
    -----
    - The function excludes sheets named "all together" and "Sources" from processing.
    - The 'survey_index' is derived from the 'Filename' column by splitting the filename at the first period.
    """

    data_sheets = pd.ExcelFile(metadata_excel).sheet_names
    data_sheets = [
        sheet for sheet in data_sheets if sheet not in ["all together", "Sources"]
    ]
    metadata = pd.concat(
        [pd.read_excel(metadata_excel, sheet_name=sheet) for sheet in data_sheets]
    )
    metadata.reset_index(drop=True, inplace=True)
    metadata["survey_index"] = metadata["Filename"].apply(lambda x: x.split(".")[0])
    metadata.set_index("survey_index", inplace=True)

    # Improve reference to positional reference systems (we assume UTM 31N here)
    # metadata["Horizontal Datum"] = metadata["Horizontal Datum"].apply(
    #     lambda x: pyproj.CRS(x + " / UTM zone 31N")
    # )
    return metadata


def nlho_from_opendap(
    url: str = r"https://opendap.deltares.nl/thredds/dodsC/opendap/hydrografie/surveys/x410000y5660000.nc",
):
    data = xr.open_dataset(url)
    return data


def get_nlho_tiles_and_files(
    nlho_grids_folder: Path | str,
) -> tuple[set[tuple[int, int]], list[Path]]:
    """
    Get the bounds of NLHO tiles and the list of .asc files from a folder.

    Parameters
    ----------
    nlho_grids_folder : Path or str
        The path to the folder containing NLHO grid files in .asc format.

    Returns
    -------
    tuple
        A tuple containing:
        - A set of tuples where each tuple contains the x and y lower left corner
          coordinates of a tile.
        - A list of Path objects representing the .asc files in the folder.
    """

    files = list(Path(nlho_grids_folder).glob("*.asc"))
    tiles = set(
        [
            (
                int(re.search(r"x(\d+)", f.stem).group(1)),
                int(re.search(r"y(\d+)", f.stem).group(1)),
            )
            for f in files
        ]
    )
    return tiles, files


def tile_surveys_to_netcdf(
    tile: tuple[int, int],
    metadata: pd.DataFrame,
    files: list[Path],
    output_folder: Path | str,
) -> None:
    """
    Export all NLHO surveys in a tile to NetCDF files.

    Parameters
    ----------
    tile : tuple of int
        A tuple containing the x and y lower left corner coordinates of the tile.
    nlho_grids_folder : Path or str
        The path to the folder containing NLHO grid files in .asc format.
    output_folder : Path or str
        The path to the folder where the NetCDF files will be saved.
    """
    tile_files = [
        f
        for f in files
        if (int(re.search(r"x(\d+)", f.stem).group(1)) == tile[0])
        & (int(re.search(r"y(\d+)", f.stem).group(1)) == tile[1])
    ]

    survey_names = [f.split("_")[2] for f in tile_files]
    tile_data = [
        rio.open_rasterio(file).squeeze().drop_vars(["band", "spatial_ref"])
        for file in tile_files
    ]

    # logic goes here
    # data = rio.open_rasterio(file).squeeze().drop_vars(["band", "spatial_ref"])
    # data.rio.write_crs(32631)
    # data.rio.write_nodata(data.rio.nodata, encoded=False, inplace=True)
    # data = data.where(data != data.rio.nodata, np.nan)
    # data.attrs["_FillValue"] = np.nan
    # data.to_netcdf(output_folder / f"{file.stem}.nc", engine="h5netcdf")


@dataclass
class NLHOAttributes:
    """
    Dataclass for NLHO survey metadata attributes.
    """

    id: str = (
        f"hydrografie_survey_grids_release_{get_current_datetime(format='%Y%m%d')}"
    )
    naming_authority: str = "deltares.nl"
    Metadata_Conventions: str = "CF-1.6"
    metadata_link: str = ""
    title: str = "Hydrografie survey grids"
    summary: str = (
        "bathymetry and topography measurements of the Dutch Continental Shelf"
    )
    keywords: str = "bathymetry, coast"
    keywords_vocabulary: str = "http://www.eionet.europa.eu/gemet"
    standard_name_vocabulary: str = (
        "http://cf-pcmdi.llnl.gov/documents/cf-standard-names"
    )
    history: str = f"Created on {get_current_datetime()} by {get_current_username()} on computer {get_computer_name()} with script {get_current_package_name()}"
    cdm_data_type: str = "grid"
    creator_name: str = "Koninklijke Marine Dienst der Hydrografie"
    creator_url: str = "www.hydro.nl"
    creator_email: str = "info@hydro.nl"
    institution: str = "Koninklijke Marine Dienst der Hydrografie"
    date_issued: str = f"{get_current_datetime()}"
    publisher_name: str = f"{get_current_username()}"
    publisher_url: str = "http://www.deltares.nl"
    publisher_email: str = "info@deltares.nl"
    processing_level: str = "final"
    WARNING: str = "THIS DATA IS NOT TO BE USED FOR NAVIGATIONAL PURPOSES. FOR NAVIGATION CHARTS PLEASE REFER TO <http://www.defensie.nl/marine/hydrografie/nautische_producten/navigatiekaarten> & <http://www.vaarweginformatie.nl>"
    license: str = "These data can be used freely for research purposes provided that the following source is acknowledged: Dienst der Hydrografie. disclaimer: This data is made available in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."
    projectioncoverage_x: list = None
    projectioncoverage_y: list = None
    date_created: str = f"{get_current_datetime()}"
    date_modified: str = f"{get_current_datetime()}"
    timecoverage: str = ""
    geospatialcoverage_northsouth: list = None
    geospatialcoverage_eastwest: list = None
    geospatial_lon_units: str = "degrees_east"
    geospatial_lon_min: float = 0.0
    geospatial_lon_max: float = 0.0
    geospatial_lat_units: str = "degrees_north"
    geospatial_lat_min: float = 0.0
    geospatial_lat_max: float = 0.0
    geospatial_vertical_units: str = "m"
    geospatial_vertical_positive: str = "up"
    geospatial_vertical_min: float = 0
    geospatial_vertical_max: float = 0
    time_coverage_units: str = ""
    source_data: str = "https://repos.deltares.nl/repos/ODyn/trunk/RawData/CorrectPointData/Mariene/, revision 47"
    processing_software: str = "https://repos.deltares.nl/repos/ODyn/trunk/Tools/Java/Sourcecode/GridSplitBatch/, revision 47"
    processing_method: str = (
        "Inverse Distance Weight interpolation of LOV2 data with radius 100 m"
    )
    DODS_strlen: int = 100
    DODS_dimName: str = "stringsize"
    DODS_EXTRA_Unlimited_Dimension: str = "time"
    EXTRA_DIMENSION_dim16: int = 16

    def __setattr__(self, name, value):
        """
        Set date_modified to current time if any attribute is changed.
        """
        if hasattr(self, name) and getattr(self, name) != value:
            super().__setattr__("date_modified", get_current_datetime())
        super().__setattr__(name, value)


if __name__ == "__main__":
    # example = nlho_from_opendap()
    tiles, files = get_nlho_tiles_and_files(r"p:\tgg-mariene-data\__UPDATES\GRIDS")
    for tile in tiles:
        tile_surveys_to_netcdf(
            tile,
            files,
            r"",
        )

    metadata = parse_survey_metadata(
        r"p:\tgg-mariene-data\__UPDATES\SVN_CHECKOUTS\metadata_SVN\METADATA_ALL.xlsx"
    )
    attrs_init = NLHOAttributes()
    attrs_init.processing_level = "intermediate"
