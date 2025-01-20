from pathlib import Path, WindowsPath

import xarray as xr

from bathyu.main import AlignedRasters, TiledRasters


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
    if isinstance(data, (AlignedRasters, TiledRasters)):
        da_to_save = data.data
    elif isinstance(data, xr.DataArray):
        da_to_save = data
    else:
        raise(TypeError("data must be an xarray.DataArray, AlignedRasters, or TiledRasters"))

    if compress:
        da_to_save.to_netcdf(
            file,
            engine="h5netcdf",
            encoding={
                da_to_save.name
                or "__xarray_dataarray_variable__": {
                    "zlib": True,
                    "complevel": compress_level,
                }
            },
        )

    else:
        da_to_save.to_netcdf(file, engine="h5netcdf")


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
    if isinstance(data, TiledRasters):
        da_to_save = data.data
    elif isinstance(data, xr.DataArray):
        da_to_save = data
    else:
        raise(TypeError("data must be an xarray.DataArray or TiledRasters"))

    if compress:
        da_to_save.rio.to_raster(file, driver="GTiff", compress="LZW")
    else:
        da_to_save.rio.to_raster(file, driver="GTiff")
