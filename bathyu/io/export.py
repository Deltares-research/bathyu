from functools import singledispatch
from pathlib import Path, WindowsPath

import xarray as xr
from dask.diagnostics import ProgressBar

from bathyu.main import AlignedRasters, TiledRasters


@singledispatch
def to_nc(
    data: xr.DataArray | AlignedRasters | TiledRasters,
    file: str | WindowsPath,
    compress=False,
    compress_level=9,
) -> None:
    """
    Export an xarray.DataArray to a (compressed) NetCDF file using zlib
    compression at the specified level.

    Parameters
    ----------
    data : xr.DataArray | AlignedRasters | TiledRasters
        An xarray.DataArray to save to NetCDF. Can also be an AlignedRasters or
        TiledRasters object.
    file : str | WindowsPath
        Path of the file to be saved.
    compress : bool, optional
        Whether to compress the output file, by default False.
    compress_level : int, optional
        Compression level to use between 1 and 9, by default 9.

    Raises
    ------
    TypeError
        If the input data is not an xarray.DataArray, AlignedRasters, or TiledRasters.
    """
    if compress:
        delayed = data.to_netcdf(
            file,
            engine="h5netcdf",
            encoding={
                data.name or "__xarray_dataarray_variable__": {
                    "zlib": True,
                    "complevel": compress_level,
                }
            },
            compute=False,
        )

    else:
        delayed = data.to_netcdf(file, engine="h5netcdf", compute=False)

    with ProgressBar():
        print("Exporting to NetCDF...")
        delayed.compute()


@singledispatch
def to_geotiff(
    data: xr.DataArray | TiledRasters, file: str | WindowsPath, compress=False
) -> None:
    """
    Export an xarray.DataArray with (y, x) dimensions to a compressed GeoTIFF file
    using LZW compression. Also works with TiledRasters objects.

    Parameters
    ----------
    data : xr.DataArray | TiledRasters
        The DataArray to save. Must have only two dimensions 'y' and 'x'.
    file : str | WindowsPath
        Path of the file to be saved.

    Raises
    ------
    TypeError
        If the input data is not an xarray.DataArray or TiledRasters
    """
    if compress:
        data.rio.to_raster(file, driver="GTiff", compress="LZW")
    else:
        data.rio.to_raster(file, driver="GTiff")


@to_nc.register
def _(
    data: AlignedRasters | TiledRasters,
    file: str | WindowsPath,
    compress=False,
    compress_level=9,
) -> None:
    """
    Implementation of to_nc for AlignedRasters and TiledRasters objects.
    """
    to_nc(data.data, file, compress=compress, compress_level=compress_level)


@to_geotiff.register
def _(
    data: TiledRasters,
    file: str | WindowsPath,
    compress=False,
) -> None:
    """
    Implementation of to_geotiff for TiledRasters objects.
    """
    to_geotiff(data.data, file, compress=compress)
