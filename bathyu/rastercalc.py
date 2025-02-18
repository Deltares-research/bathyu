from functools import singledispatch

import dask.array as da
import numpy as np
import xarray as xr
from bottleneck import push
from dask import delayed


@singledispatch
def most_recent(
    array: np.ndarray | xr.DataArray, axis_or_dim
) -> np.ndarray | xr.DataArray:
    """
    Get the last non-NaN value for each cell along an axis. This essentially
    creates a mosaic of the data with the most recent measurement along the specified
    axis being cast to the output array. This operation reduces the dimensionality
    of the input array by one.

    Parameters
    ----------
    array : np.ndarray or xr.DataArray
        The input array from which to extract the most recent non-NaN values.
    axis_or_dim : int or str
        The axis along which to perform the operation. For xarray.DataArray, this can
        be a dimension name.

    Returns
    -------
    np.ndarray or xr.DataArray
        An array with the most recent non-NaN values along the specified axis.
    """
    raise NotImplementedError(f"most_recent not implemented for {type(array)}")


@most_recent.register
def _(array: xr.DataArray, dim) -> xr.DataArray:
    if isinstance(dim, int):
        dim = array.dims[dim]
    ffilled = array.ffill(dim=dim)
    last = ffilled[-1]
    return last


@most_recent.register
def _(array: np.ndarray, axis) -> np.ndarray:
    ffilled = push(array, axis=axis)
    last = ffilled.take(indices=-1, axis=axis)
    return last


def slope(array: np.ndarray) -> np.ndarray:
    """
    Calculate the slope of a 2D array using the gradient along both axes.

    Parameters
    ----------
    array : np.ndarray
        A 2D numpy array representing the input data.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the calculated slope.

    Notes
    -----
    The slope is calculated as the absolute value of the product of the gradients along
    the x and y axes.
    """
    slopex = np.gradient(array, axis=1)
    slopey = np.gradient(array, axis=2)
    slope = np.abs(slopex * slopey)
    return slope


def differences_along_time(array: np.ndarray) -> np.ndarray:
    """
    Calculate the differences along the last axis of the input array.

    Parameters
    ----------
    array : np.ndarray
        Input array for which the differences along the last axis are to be calculated.

    Returns
    -------
    np.ndarray
        An array of the same shape as `array` with the differences along the last axis.
    """
    diff = np.diff(array, axis=-1)
    return diff


def time_gradient(array: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Calculate the time gradient of a 3D array. The time-axis is assumed to be the first
    axis (0), but can be adjusted using the `axis` parameter.

    Parameters
    ----------
    array : np.ndarray
        A 3D numpy array where the first dimension represents time.
    axis : int, optional
        The axis along which the gradient is to be calculated. Default is 0.

    Returns
    -------
    np.ndarray
        A 3D numpy array representing the gradient along the time axis.
    """
    array = array.rechunk([4, 1000, 1000])
    slopet = np.gradient(array, axis=axis)
    return slopet


def fill_with_index(array: np.ndarray) -> np.ndarray:
    """
    Fill a 3D array with increasing index values along the first axis.

    Parameters
    ----------
    array : np.ndarray
        A 3D numpy array.

    Returns
    -------
    np.ndarray
        A new 3D numpy array where each slice along the first axis is filled
        with increasing index values starting from 1. The shape of the returned
        array is the same as the input array. Non-finite values in the input
        array are replaced with NaN in the output array.
    """
    shape = array.shape
    increasing_values = np.arange(1, shape[0] + 1)
    new_array = increasing_values[:, np.newaxis, np.newaxis]
    new_array = np.tile(new_array, (1, shape[1], shape[2]))
    new_array = new_array * np.where(np.isfinite(array), 1, np.nan)
    return new_array


def cell_coverage(array: np.ndarray, axis: int) -> np.ndarray:
    """
    Calculate the number of non-NaN elements along a specified axis.

    Parameters
    ----------
    array : np.ndarray
        Input array containing NaN values.
    axis : int
        Axis along which to count non-NaN elements.

    Returns
    -------
    np.ndarray
        An array with the count of non-NaN elements along the specified axis.
    """
    return np.count_nonzero(~np.isnan(array), axis=axis)


def reindex_chunks(chunk: np.ndarray, **kwargs) -> np.ndarray:
    """
    Reindex a chunk of a raster array to match the resolution and grid of another raster.
    This function is called by `dask.array.map_blocks` to reindex each chunk to match the
    resolution and grid of the reference raster (kwargs["raster"]).

    Parameters
    ----------
    chunk : np.ndarray
        The input chunk of the raster array to be reindexed.
    **kwargs : dict
        Additional keyword arguments:
        - raster : xarray.DataArray
            The reference raster to reindex against.
        - resolution : float
            The resolution tolerance for the reindexing process.

    Returns
    -------
    np.ndarray
        The reindexed chunk of the raster array.
    """
    chunk = kwargs["raster"].reindex_like(
        chunk,
        method="nearest",
        tolerance=kwargs["resolution"],
        fill_value=np.nan,
    )
    return chunk
