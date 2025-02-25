import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as rio
import xarray as xr
from pyproj import CRS

from bathyu.io.export import to_nc
from bathyu.nlho.attributes import (
    MetaAttrs,
    NlhoGlobalAttributes,
    TimeAttrs,
    XAttrs,
    YAttrs,
    ZAttrs,
)
from bathyu.rastercalc import cell_coverage, fill_with_index, most_recent
from bathyu.utils import set_da_attributes


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
    metadata["survey_index"] = metadata["Filename"].apply(
        lambda x: x.split(".")[0].strip()
    )
    metadata.set_index("survey_index", inplace=True)

    return metadata


def nlho_from_opendap(
    url: str = r"https://opendap.deltares.nl/thredds/dodsC/opendap/hydrografie/surveys/x410000y5660000.nc",
    **kwargs,
):
    data = xr.open_dataset(url, **kwargs)
    return data


def get_nlho_tiles_and_files(
    nlho_grids_folder: Path | str, bbox: list = None
) -> tuple[set[tuple[int, int]], list[Path]]:
    """
    Get the bounds of NLHO tiles and the list of .asc files from a folder.

    Parameters
    ----------
    nlho_grids_folder : Path or str
        The path to the folder containing NLHO grid files in .asc format.
    bbox : list, optional
        A bounding box specified as [min_x, min_y, max_x, max_y] to filter the tiles.

    Returns
    -------
    tuple
        A tuple containing:
        - A set of tuples where each tuple contains the x and y lower left corner
          coordinates of a tile.
        - A list of Path objects representing the .asc files in the folder.
    """

    files = list(Path(nlho_grids_folder).glob("*.asc"))

    if bbox:
        tiles = set(
            [
                (
                    int(re.search(r"x(\d+)", f.stem).group(1)),
                    int(re.search(r"y(\d+)", f.stem).group(1)),
                )
                for f in files
                if (int(re.search(r"x(\d+)", f.stem).group(1)) >= bbox[0])
                & (int(re.search(r"x(\d+)", f.stem).group(1)) < bbox[2])
                & (int(re.search(r"y(\d+)", f.stem).group(1)) >= bbox[1])
                & (int(re.search(r"y(\d+)", f.stem).group(1)) < bbox[3])
            ]
        )
    else:
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


def validate_nlho_metadata(metadata: pd.DataFrame, files) -> pd.DataFrame:
    """
    Validate and update the metadata DataFrame based on the survey names found in the
    grid filenames that are not present in the metadata. This is typically because the
    survey name in the filename does not exactly match the survey name in the metadata.

    Parameters
    ----------
    metadata : pd.DataFrame
        A DataFrame containing metadata with survey names as the index.
    files : list
        A list of ascii files containing the NLHO grid data.

    Returns
    -------
    pd.DataFrame
        The updated metadata DataFrame with missing survey names added if found.

    Notes
    -----
    - The function extracts survey names from the filenames and compares them with the survey names in the metadata index.
    - If there are survey names in the filenames that are not present in the metadata, the function attempts to find and update the missing metadata.
    - The survey names in the filenames are expected to be in the format where the survey name is the third element when split by underscores.
    """

    unique_surveys_from_files = set([f.name.split("_")[2] for f in files])
    unique_surveys_from_metadata = set(metadata.index)
    missing_surveys_in_metadata = (
        unique_surveys_from_files - unique_surveys_from_metadata
    )
    if missing_surveys_in_metadata:
        print(
            f"Survey names that are mentioned in filenames, but absent in metadata:\n\n{missing_surveys_in_metadata}\n\n   >> Trying to find the missing metadata from the metadata file."
        )
        missing_survey_numbers = [
            re.search(r"\d+", survey).group() for survey in missing_surveys_in_metadata
        ]
        survey_number_to_id = {
            re.search(r"\d+", survey).group(): survey for survey in metadata.index
        }
        new_index_mapping = {
            survey_number_to_id[missing_survey_number]: missing_survey_in_metadata
            for missing_survey_number, missing_survey_in_metadata in zip(
                missing_survey_numbers, missing_surveys_in_metadata
            )
        }
        print(
            f"\n\nFound the following matches:\n\n{new_index_mapping}\n\nUpdating metadata indices based on the found matches."
        )

        # Add the new metadata to the existing metadata, not replacing the existing
        # metadata, but through adding new rows as some surveys may be referenced
        # by multiple names (this dataset is a bit of a mess)
        extra_rows = metadata.loc[new_index_mapping.keys()]
        extra_rows.rename(index=new_index_mapping, inplace=True)
        metadata = pd.concat([metadata, extra_rows])
    return metadata


def extract_isource_and_coverage(
    data: xr.Dataset, tile_files: pd.DataFrame, survey_names: list[str]
) -> xr.DataArray:
    """
    Extract time-varying data variables from a data array.

    Parameters
    ----------
    data : xr.DataArray
        The data array containing the time dimension.

    Returns
    -------
    xr.DataArray
        A new data array containing only the time-varying data variables.

    Notes
    -----
    - The function assumes that the time dimension is named 'time'.
    - The function extracts all variables that are not coordinates or attributes.
    """
    coverage = cell_coverage(data.z.values, axis=0).astype(np.float32)
    isource = fill_with_index(data.z.values)
    mosaic = most_recent(data.z.values, 0)

    data = data.assign(
        **{
            "mosaic": (("y", "x"), mosaic),
            "isource": (("time", "y", "x"), isource),
            "coverage": (("y", "x"), coverage),
        }
    )

    data["mosaic"] = data["mosaic"].assign_attrs(data["z"].attrs)
    data["mosaic"].attrs.update(
        **{
            "standard_name": "altitude",
            "long_name": "Mosaic of surveys",
            "definition": "Mosaic of surveys showing the most recent data for each cell",
            "actual_range": (
                np.float32(data.mosaic.min().item()),
                np.float32(data.mosaic.max().item()),
            ),
        }
    )
    isource_flags = " ".join([str(float(r + 1)) for r in range(len(survey_names))])
    isource_flag_meanings = " ".join([f.name for f in tile_files])
    data["isource"] = data["isource"].assign_attrs(
        **{
            "long_name": "source file index",
            "definition": "zero based index of source file. The given index in a time slice corresponds to the survey that is most recent for that location and time.",
            "flag_values": isource_flags,
            "flag_meanings": isource_flag_meanings,
            "grid_mapping": "crs",
        }
    )
    data["coverage"] = data["coverage"].assign_attrs(data["z"].attrs)
    data["coverage"].attrs.update(
        **{
            "standard_name": "number_of_observations",
            "long_name": "Survey coverage",
            "definition": "Number of surveys covering each cell in the time dimension",
            "units": "1",
            "actual_range": (
                np.float32(data.coverage.min().item()),
                np.float32(data.coverage.max().item()),
            ),
        }
    )
    return data


def tile_surveys_to_netcdf(
    tile: tuple[int, int],
    files: list[Path],
    metadata_data: pd.DataFrame,
    metadata_fields: dict,
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

    survey_names = [f.name.split("_")[2] for f in tile_files]
    survey_times = [
        metadata_data.loc[survey, "Survey End Date"] for survey in survey_names
    ]

    # Sort tile_files, survey_names an survey_times based on survey_times (old to new)
    tile_files = [file for _, file in sorted(zip(survey_times, tile_files))]
    survey_names = [f.name.split("_")[2] for f in tile_files]
    survey_times.sort()

    # Convert survey times to days since 1970-01-01
    survey_times = pd.to_datetime(survey_times)
    survey_times = np.int32(
        (survey_times - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D")
    )

    # Get raster data from tile files and concatenate
    tile_data = [
        rio.open_rasterio(file).squeeze().drop_vars(["band", "spatial_ref"])
        for file in tile_files
    ]
    concatenated_surveys = xr.concat(tile_data, dim="time")
    concatenated_surveys = concatenated_surveys.assign_coords(
        time=("time", survey_times)
    )

    # Replace fill values with NaN
    concatenated_surveys = concatenated_surveys.where(
        concatenated_surveys != concatenated_surveys.attrs["_FillValue"], np.nan
    )
    concatenated_surveys.rio.update_attrs({"_FillValue": np.nan}, inplace=True)

    # Set attributes for data and coordinate data_vars (z, x, y and time)
    concatenated_surveys = concatenated_surveys.to_dataset(name="z")
    concatenated_surveys["z"] = concatenated_surveys["z"].assign_attrs(
        ZAttrs.from_dataset(concatenated_surveys).as_dict
    )
    concatenated_surveys["x"] = concatenated_surveys["x"].assign_attrs(
        XAttrs.from_dataset(concatenated_surveys).as_dict
    )
    concatenated_surveys["y"] = concatenated_surveys["y"].assign_attrs(
        YAttrs.from_dataset(concatenated_surveys).as_dict
    )
    concatenated_surveys["time"] = concatenated_surveys["time"].assign_attrs(
        TimeAttrs.from_dataset(concatenated_surveys).as_dict
    )

    # Add isource and coverage data variables along with their attributes
    concatenated_surveys = extract_isource_and_coverage(
        concatenated_surveys, tile_files, survey_names
    )

    # Set the CRS of the dataset to EPSG:32631 following CF conventions
    concatenated_surveys["crs"] = xr.DataArray(
        np.array(32631, dtype=np.int32), attrs=CRS.from_epsg(32631).to_cf()
    )

    # Set other metadata and corresponding attributes
    for metadata in metadata_fields.values():
        metadata_object = MetaAttrs.from_metadata_df(
            metadata_data, survey_names, **metadata
        )
        concatenated_surveys = concatenated_surveys.assign(
            **{metadata_object.var_name: ("time", metadata_object.values)}
        )
        concatenated_surveys[metadata_object.var_name] = concatenated_surveys[
            metadata_object.var_name
        ].assign_attrs(metadata_object.attrs_as_dict)
        if "timeunits" in metadata_object.attrs_as_dict.keys():
            concatenated_surveys[metadata_object.var_name].encoding["units"] = (
                metadata_object.timeunits
            )

    # Set global attributes
    concatenated_surveys = set_da_attributes(
        concatenated_surveys,
        **NlhoGlobalAttributes.from_dataset(concatenated_surveys).as_dict,
    )

    # Export to NetCDF
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"x{tile[0]}y{tile[1]}.nc"
    to_nc(concatenated_surveys, output_file, compress=True)


def combine_nlho_mosaics(nc_files):
    """
    Combine NLHO mosaics from NetCDF files in a folder.

    Parameters
    ----------
    nc_files: Path or str
        The path to the folder containing NLHO NetCDF files.

    Returns
    -------
    xr.Dataset
        A dataset containing the combined mosaics from all NetCDF files in the folder.
    """
    datasets = [xr.open_dataset(file).mosaic for file in nc_files]
    combined = xr.combine_by_coords(datasets, combine_attrs="override")
    combined['x'] = combined['x'].assign_attrs(XAttrs.from_dataset(combined).as_dict)
    combined['y'] = combined['y'].assign_attrs(YAttrs.from_dataset(combined).as_dict)
    return combined
