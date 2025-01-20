import re
from pathlib import Path, WindowsPath
from typing import Iterable

import dask.array as da
import geopandas as gpd
import numpy as np
import rioxarray as rio
import xarray as xr
from rioxarray.merge import merge_arrays

from bathyu import rastercalc, utils


class AlignedRasters:
    """
    The AlignedRasters class revolves around an Xarray dataarray with time, x and y
    coordinates and allows users to apply its methods to the dataarray such as minumum,
    maximum, most recent/oldest mosaic, dz/dt etc. Results can be written to a
    compressed NetCDF (for results with time, x, y coordinates) or to a compressed
    GeoTIFF (for results reduced along the time axis, so with only x, y coordinates).
    """

    def __init__(
        self,
        data: str | WindowsPath | xr.DataArray,
        chunk_params: dict = {"time": 1, "x": 2000, "y": 2000},
    ):
        """
        Initialize AlignedRasters either from an xarray.DataArray of already aligned
        rasters with dimensions (time, y, x). E.g. when you have previously saved
        some aligned rasters in a NetCDF file. Otherwise use the 'from_files' class
        method to construct the object from multiple (not-yet-aligned) raster files.

        Parameters
        ----------
        data : str | WindowsPath | xarray.DataArray
            Can be a path pointing to a NetCDF file of aligned rasters with dimensions
            (time, y, x) or is already an xarray.DataArray instance.
        chunk_params : dict
            Dictionary of chunk parameters. Must be of the form:
            {'time': int, 'x': int, 'y': int}
        """
        # Get chunk attributes
        self.chunk_3d = chunk_params
        self.chunk_2d = chunk_params.copy().__delitem__("time")

        # Assign data attribute
        if isinstance(data, (str, WindowsPath)):
            data = xr.open_dataarray(data)
        self.data = data.chunk(self.chunk_3d)

    @classmethod
    def from_files(
        cls,
        files: Iterable[str | WindowsPath],
        datetime_format: str,
        resolution: float | int | str = "minimum",
        bounds: str = "inferred",
        bbox: list[int | float] = [],
        desired_crs: str = "epsg:28992",
        dtype: np.dtype = np.float32,
        chunk_params: dict = {"time": 1, "x": 2000, "y": 2000},
        force_nodata: bool = False,
        scale_factors: float | list[float] = None,
    ):
        """
        Create aligned rasters from multiple raster files.

        Parameters
        ----------
        files : Iterable[str  |  WindowsPath]
            Iterable of paths to raster files.
        datetime_format : str
            Format of the date/time indication in filenames. Uses Python datetime
            format, see: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
        resolution : float | int | str, optional
            Desired resolution of aligned DataArray. Either "minimum", "maximum" or a
            number. "minimum" will use the resolution of the lowest resolution input
            raster, whereas "maximum" will use the best resolution found among input
            rasters. Alternatively can be user-determined by giving the desired cell
            size as a float or integer number. By default "minimum".
        bounds : str, optional
            Either "inferred" or "user". "inferred" means that the maximum bounds of all
            input rasters will be used whereas the option "user" requires the bbox
            keyword argument with [xmin, xmax, ymin, ymax] given. By default "inferred".
        bbox : list[int  |  float], optional
            If the argument bounds is "user", the bbox will contain
            [xmin, xmax, ymin, ymax] to be used for the aligned DataArray. By default []
        desired_crs : str, optional
            Coordinate reference system to use, by default "epsg:28992".
        dtype : numpy.dtype, optional
            Datatype to use for raster values, by default numpy.float32.
        chunk_params : dict
            Dictionary of chunk parameters. Must be of the form:
            {'time': int, 'x': int, 'y': int}.
        force_nodata : bool, optional
            Whether to force the nodata value to np.nan. By default False.
        scale_factors : None | list[float], optional
            Scale factor to apply to each raster. By default None. Must have the same
            length as the amount of input rasters.

        Returns
        -------
        Instance of class AlignedRasters
        """
        # Read files and get datetimes
        files = list(files)
        times = [
            utils.find_date_in_filename_from_format(f, datetime_format) for f in files
        ]
        rasters = [
            rio.open_rasterio(f).squeeze().drop_vars(["band", "spatial_ref"])
            for f in files
        ]

        # Determine total bounds and resolution
        if bounds == "inferred":
            bounds = np.array([r.rio.bounds() for r in rasters])
            xmin = np.floor(bounds[:, 0].min())
            ymin = np.floor(bounds[:, 1].min())
            xmax = np.ceil(bounds[:, 2].max())
            ymax = np.ceil(bounds[:, 3].max())
        elif bounds == "user" and len(bbox) == 4:
            xmin = bbox[0]
            xmax = bbox[1]
            ymin = bbox[2]
            ymax = bbox[3]
            rasters = [r.sel(x=slice(xmin, xmax), y=slice(ymax, ymin)) for r in rasters]
        else:
            raise TypeError(
                "Invalid bounds. Check the validity of your bounds and bbox arguments!"
            )

        resolutions = np.array([r.rio.resolution() for r in rasters])
        if resolution == "minimum":
            resolution = np.abs(resolutions).max()
        elif resolution == "maximum":
            resolution = np.abs(resolutions).min()
        elif isinstance(resolution, (int, float)):
            resolution = resolution

        # Create target coordinate labels, grid and chunked dataarray
        x_coordinates = np.arange(xmin, xmax, resolution, dtype=dtype)
        y_coordinates = np.arange(ymax, ymin, -resolution, dtype=dtype)
        target_single_grid = xr.DataArray(
            data=da.empty(
                [len(y_coordinates), len(x_coordinates)],
                chunks=(chunk_params["y"], chunk_params["x"]),
                dtype=dtype,
            ),
            coords={"y": y_coordinates, "x": x_coordinates},
        )
        result_da = xr.DataArray(
            data=da.empty(
                [len(times), len(y_coordinates), len(x_coordinates)],
                chunks=(chunk_params["time"], chunk_params["y"], chunk_params["x"]),
                dtype=dtype,
            ),
            coords={"time": times, "y": y_coordinates, "x": x_coordinates},
        )

        # Reproject input rasters to target grid
        for i, (raster, time) in enumerate(zip(rasters, times)):
            if force_nodata:
                raster = raster.where(raster != raster.attrs["_FillValue"], np.nan)
                raster.rio.update_attrs({"_FillValue": np.nan})
            if scale_factors:
                raster = raster * scale_factors[i]
            print("Reprojecting to new grid")
            result_da.loc[dict(time=time)] = target_single_grid.map_blocks(
                rastercalc.reindex_chunks,
                kwargs={"raster": raster, "resolution": resolution},
                template=target_single_grid,
            )
        result_da = result_da.where(result_da != np.nan)
        result_da.attrs["_FillValue"] = np.nan
        result_da.rio.write_crs(desired_crs)
        result_da = utils.set_da_attributes(result_da)

        return cls(result_da, chunk_params)

    def minimum(self) -> xr.DataArray:
        """
        Get the minimum value for each cell along the time axis. This operation reduces
        aligned raster data along the 'time' axis.

        Returns
        -------
        xarray.DataArray
            DataArray with dimensions (y, x).
        """
        da_minimum = self.data.min(dim="time")
        return da_minimum

    def maximum(self) -> xr.DataArray:
        """
        Get the maximum value for each cell along the time axis. This operation reduces
        aligned raster data along the 'time' axis.

        Returns
        -------
        xarray.DataArray
            DataArray with dimensions (y, x).
        """
        da_maximum = self.data.max(dim="time")
        return da_maximum

    def most_recent(self) -> xr.DataArray:
        """
        Get the last non-nan value for each cell along the time axis. This essentially
        creates a mosaic of the data with the most recent measurement along the time
        axis being cast to the output array. This operation reduces aligned raster data
        along the 'time' axis.

        Returns
        -------
        xarray.DataArray
            DataArray with dimensions (y, x).
        """
        ffilled = self.data.ffill(dim="time")
        last = ffilled[-1]
        return last

    def most_recent_index(self) -> xr.DataArray:
        """
        Get the last index with a non-nan value for each cell along the time axis. This
        generates a raster of integers that indicates which index along the time axis
        holds the most recent valid value. This operation reduces aligned raster data
        along the 'time' axis.

        Returns
        -------
        xarray.DataArray
            DataArray with dimensions (y, x).
        """
        index_da = xr.apply_ufunc(rastercalc.fill_with_index, self.data, dask="allowed")
        ffilled = index_da.ffill(dim="time")
        last_index = ffilled[-1]
        return last_index

    def oldest(self) -> xr.DataArray:
        """
        Get the first non-nan value for each cell along the time axis. This essentially
        creates a mosaic of the data with the oldest measurement along the time axis
        being cast to the output array. This operation reduces aligned raster data
        along the 'time' axis.

        Returns
        -------
        xarray.DataArray
            DataArray with dimensions (y, x).
        """
        bfilled = self.data.bfill(dim="time")
        first = bfilled[0]
        return first

    def oldest_index(self) -> xr.DataArray:
        """
        Get the first index with a non-nan value for each cell along the time axis. This
        generates a raster of integers that indicates which index along the time axis
        holds the oldest valid value. This operation reduces aligned raster data along
        the 'time' axis.

        Returns
        -------
        xarray.DataArray
            DataArray with dimensions (y, x).
        """
        index_da = xr.apply_ufunc(rastercalc.fill_with_index, self.data, dask="allowed")
        bfilled = index_da.bfill(dim="time")
        first_index = bfilled[0]
        return first_index

    def coverage_number(self) -> xr.DataArray:
        """
        Get the amount of raster coverage for each cell. The output is a DataArray of
        integers indicating the number of aligned rasters that cover a cell. This
        operation reduces aligned raster data along the 'time' axis.

        Returns
        -------
        xarray.DataArray
            DataArray with dimensions (y, x).
        """
        coverage = self.data.reduce(rastercalc.cell_coverage, axis=0)
        return coverage.compute()

    def slope(self):
        """
        Compute the slope maps for each of the aligned rasters. No reduction takes
        place, so the result is a new instance of AlignedRasters with
        AlignedRasters.data being an xarray.DataArray with the same (time, y, x)
        dimensions, but the values being the first-derivative of the input surfaces.

        Returns
        -------
        AlignedRasters
            New instance of AlignedRasters, but with the first-derivative of the input.
        """
        slopes = xr.apply_ufunc(rastercalc.slope, self.data, dask="allowed")
        slopes = utils.set_da_attributes(slopes)
        return self.__class__(slopes)

    def differences(self):
        """
        Compute the differences between each array in self.data along the 'time' dimension.

        Returns
        -------
        AlignedRasters
            New instance of AlignedRasters with the differences computed along the 'time'
            dimension. The resulting DataArray has the same dimensions as the input, but
            with the first value removed along the 'time' dimension.
        """
        diff = xr.apply_ufunc(
            rastercalc.differences_along_time,
            self.data,
            dask="allowed",
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            exclude_dims=set(("time",)),
            vectorize=True,
        )
        diff = diff.assign_coords(time=self.data.time[1:])
        diff = diff.transpose("time", "y", "x")
        diff = utils.set_da_attributes(diff)
        return self.__class__(diff)

    def dzdt(self):
        """
        Calculate the time gradient of the data using the `time_gradient` function.
        This method applies the `time_gradient` function to the data stored in the
        object using xarray's `apply_ufunc` method, allowing for Dask parallelization.

        Returns:
            An instance of the same class with the time gradient applied to the data.
        """
        dzdt = xr.apply_ufunc(rastercalc.time_gradient, self.data, dask="allowed")
        time_grad = (
            np.gradient(self.data.time.values).astype("timedelta64[D]").astype(float)
        )
        time_grad_da = xr.DataArray(
            time_grad, coords={"time": self.data.time}, dims="time"
        )
        dzdt = dzdt / time_grad_da
        dzdt = utils.set_da_attributes(dzdt)
        return self.__class__(dzdt)

    def sample_from_lines(self, line_gdf: gpd.GeoDataFrame):
        # Get x and y coordinates of every point along line_gdf, spaced 1 m apart
        if not self.data.rio.crs:
            self.data = self.data.rio.write_crs("epsg:28992")

        line_gdf = line_gdf.to_crs(self.data.rio.crs)
        points = []
        for line in line_gdf.geometry:
            distance = np.arange(0, line.length, 0.1)
            points.extend([line.interpolate(d) for d in distance])

        x_coords = xr.DataArray([point.x for point in points])
        y_coords = xr.DataArray([point.y for point in points])
        sampled_data = self.data.sel(x=x_coords, y=y_coords, method="nearest")
        return sampled_data


class TiledRasters:  # pragma: no cover
    """
    A class to handle tiled raster data.

    Attributes
    ----------
    data : xr.DataArray
        The combined raster data.
    """

    def __init__(self, data: xr.DataArray):
        self.data = data
        self.data.where(self.data != self.data.rio.nodata, np.nan)
        self.data["_FillValue"] = np.nan

    @classmethod
    def from_tiled_raster_files(cls, files: Iterable[str | WindowsPath]):
        data = [
            rio.open_rasterio(f).squeeze().drop_vars(["band", "spatial_ref"])
            for f in files
        ]
        data_combined = xr.combine_by_coords(data)
        return cls(data_combined)

    @classmethod
    def from_nlho_grid_files(
        cls, file_folder: str | WindowsPath, xmin: int, ymin: int, xmax: int, ymax: int
    ):
        files = list(Path(file_folder).glob("*.asc"))

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

        # Iterate over tiles and create mosaic for each tile
        mosaics = []
        for tile in tiles:
            print(f"Processing tile: {tile}")
            tile_files = [
                f
                for f in selected_files
                if tile_search_pattern.search(f.stem).group(1) == tile
            ]

            # Make sure the files are sorted by survey number
            tile_survey_numbers = np.array(
                [
                    int(re.search(r"(\d+)", f.stem.split("_")[2]).group(1))
                    for f in tile_files
                ]
            )
            sorted_indices = np.argsort(tile_survey_numbers)
            sorted_tile_files = [tile_files[i] for i in sorted_indices]

            # Read and combine grids for the current tile
            tile_data = [
                rio.open_rasterio(f)
                .squeeze()
                .drop_vars(["band", "spatial_ref"])
                .rio.write_crs(32631)
                for f in sorted_tile_files
            ]
            mosaic = merge_arrays(tile_data, nodata=np.nan, method="last")
            mosaics.append(mosaic)

        # Combine all mosaics into one DataArray
        print("Combining tile mosaics")
        data_combined = xr.combine_by_coords(mosaics)

        return cls(data_combined)
