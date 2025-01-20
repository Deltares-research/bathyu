import pathlib
import re
from datetime import datetime
from pathlib import Path

import numpy as np


def find_date_in_filename_from_format(filename, time_format):
    """
    Extracts a date from a filename based on a given date format.

    Parameters
    ----------
    filename : Path
        The filename from which to extract the date. It should be a Path object.
    time_format : str
        The date format to use for extracting the date. This should be a format string

    Returns
    -------
    datetime
        The extracted date if found.

    Raises
    ------
    ValueError
        If no date in the given format is found in the filename.
    """
    filename_parts = re.split("_|-", filename.stem)
    time = None
    for filename_part in filename_parts:
        try:
            time = datetime.strptime(filename_part, time_format)
        except ValueError:
            continue
        if time is not None:
            return time
    else:
        raise ValueError(
            f"No date in given format was found in file: '{filename.stem}'"
        )


def find_xyz_sep(
    xyz_file, seps=["\t", "    ", "   ", "  ", " ", "|", ",", ";", "-", "_"]
):
    """
    Determines the separator used in an XYZ file.

    Parameters
    ----------
    xyz_file : str
        Path to the XYZ file.
    seps : list of str, optional
        List of potential separators to check for in the file. Default is a list of
        common separators, but others can be provided.

    Returns
    -------
    str
        The detected separator used in the XYZ file.

    Notes
    -----
    This function reads the first two lines of the file and checks for the presence of
    any of the provided separators. If a mix of separators is used in the file, the
    function may not work as expected.
    """
    with open(xyz_file) as file:
        for i in range(2):
            line = file.readline()
        sep = []
        [sep.append(char) for char in line if char in seps]
        sep = sep[1]
    return sep


def remove_df_header(df):
    """
    Removes the header from a DataFrame and converts the remaining data to float32.

    This function iterates over the DataFrame rows and checks if the diagonal element
    can be converted to a float. If a float is found, it slices the DataFrame from that
    row onwards and converts the data type to float32.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame from which the header is to be removed.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with the header removed and data type converted to float32.

    Notes
    -----
    - The function assumes that the header is located in the diagonal elements of the DataFrame.
    - If no float is found in the diagonal elements, the original DataFrame is returned.
    - The function modifies the DataFrame in place and returns the modified DataFrame.
    """
    for i in range(len(df)):
        entry = df.iloc[[i], [i]].values[0][0]
        try:
            entry = float(entry)
        except ValueError:
            continue
        if isinstance(entry, float):
            df = df[i:]
            df = df.astype(np.float32)
            return df


def find_all_files_with_suffix(folder, suffix):
    """
    Recursively find all files with a given suffix in a folder.

    Parameters
    ----------
    folder : str
        The path to the folder where the search will be performed.
    suffix : str
        The file suffix to search for (e.g., '.txt').

    Yields
    ------
    pathlib.Path
        Generator of paths to the files that match the given suffix.
    """
    for path in pathlib.Path(folder).rglob("*"):
        if path.is_file():
            if path.suffix == suffix:
                yield path


def rmtree(f: Path):
    """
    Recursively remove a directory tree.

    Parameters
    ----------
    f : Path
        The path to the directory or file to be removed.

    Notes
    -----
    This function will delete all files and directories within the specified
    directory, including the directory itself. Use with caution.
    """
    if f.is_file():
        f.unlink()
    else:
        for child in f.iterdir():
            rmtree(child)
        f.rmdir()


def set_da_attributes(da, **kwargs):
    """
    Set default spatial attributes for a DataArray object to ensure it is compatible in
    GIS software when exported as NetCDF file. Any additional attributes provided via
    kwargs will also be assigned to the DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        The DataArray to which attributes will be added.
    **kwargs : dict
        Additional attributes to assign to the DataArray.

    Returns
    -------
    xarray.DataArray
        The DataArray with updated attributes.

    Notes
    -----
    This function sets the following default attributes:
    - "_FillValue": np.nan
    - "nodatavals": np.nan
    - "x" axis attribute: "X"
    - "y" axis attribute: "Y"
    """
    da.attrs["_FillValue"] = np.nan
    da.attrs["nodatavals"] = np.nan
    da["x"].attrs["axis"] = "X"
    da["y"].attrs["axis"] = "Y"
    da = da.assign_attrs(kwargs)
    return da
