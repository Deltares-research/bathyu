from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from pyproj import CRS

from bathyu.projections import xy_to_ll
from bathyu.utils import (
    get_computer_name,
    get_current_datetime,
    get_current_package_name,
    get_current_username,
)


class AbstractAttributes(ABC):
    @classmethod
    @abstractmethod
    def from_dataset(cls, dataset):
        pass

    @property
    @abstractmethod
    def as_dict(self):
        pass


class AbstractMetadataAttributes(ABC):
    @classmethod
    @abstractmethod
    def from_metadata_df(cls, metadata_df, indices):
        pass

    @property
    @abstractmethod
    def attrs_as_dict(self):
        pass


@dataclass
class XAttrs(AbstractAttributes):
    """
    Dataclass for X-coordinate attributes.
    """

    standard_name: str = "projection_x_coordinate"
    long_name: str = "x-coordinate"
    grid_mapping: str = "crs"
    units: str = "m"
    axis: str = "X"
    definition: str = "The projection x-coordinate of the grid cell center. Refer to the global attribute 'crs' for the coordinate reference system."
    actual_range: tuple = None
    resolution: float = 25.0

    @classmethod
    def from_dataset(cls, dataset):
        """
        Create an XAttrs object from an xarray.Dataset. This method sets the
        attributes based on the dataset's coordinates and values.

        Note: the datasetmust have 'x', 'y', and 'time' coordinates.
        """
        attrs = cls()
        attrs.resolution = float(dataset.x[1] - dataset.x[0])
        attrs.actual_range = (float(dataset.x.min()), float(dataset.x.max()))
        return attrs

    @property
    def as_dict(self):
        return self.__dict__


@dataclass
class YAttrs(AbstractAttributes):
    """
    Dataclass for Y-coordinate attributes.
    """

    standard_name: str = "projection_y_coordinate"
    long_name: str = "y-coordinate"
    grid_mapping: str = "crs"
    units: str = "m"
    axis: str = "Y"
    definition: str = "The projection y-coordinate of the grid cell center. Refer to the global attribute 'crs' for the coordinate reference system."
    actual_range: tuple = None
    resolution: float = 25.0

    @classmethod
    def from_dataset(cls, dataset):
        """
        Create an YAttrs object from an xarray.Dataset. This method sets the
        attributes based on the dataset's coordinates and values.

        Note: the dataset must have 'x', 'y', and 'time' coordinates.
        """
        attrs = cls()
        attrs.resolution = float(dataset.y[1] - dataset.y[0])
        attrs.actual_range = (float(dataset.y.min()), float(dataset.y.max()))
        return attrs

    @property
    def as_dict(self):
        return self.__dict__


@dataclass
class TimeAttrs(AbstractAttributes):
    """
    Dataclass for time-coordinate attributes.
    """

    standard_name: str = "time"
    long_name: str = "Date and time at the completion of the survey"
    units: str = "days since 1970-01-01 00:00:00"
    calendar: str = "gregorian"
    axis: str = "T"
    definition: str = "Date and time at the completion of a survey in days since 1970-01-01. The time coordinate is used to represent the time of the survey."
    actual_range: tuple = None

    @classmethod
    def from_dataset(cls, dataset):
        """
        Create an ZAttrs object from an xarray.Dataset. This method sets the
        attributes based on the dataset's coordinates and values.

        Note: the dataset must have 'x', 'y', and 'time' coordinates.
        """
        attrs = cls()
        attrs.actual_range = (
            np.int32(dataset.time.min()),
            np.int32(dataset.time.max()),
        )
        return attrs

    @property
    def as_dict(self):
        return self.__dict__


@dataclass
class ZAttrs(AbstractAttributes):
    """
    Dataclass for Z-coordinate attributes.
    """

    standard_name: str = "altitude"
    long_name: str = "altitude"
    grid_mapping: str = "crs"
    units: str = "m"
    definition: str = "Lowest Astronomical Tide (LAT), the lowest tide level which can be predicted to occur under average meteorological conditions and under any combination of astronomical conditions."
    actual_range: tuple = None

    @classmethod
    def from_dataset(cls, dataset):
        """
        Create an ZAttrs object from an xarray.Dataset. This method sets the
        attributes based on the dataset's coordinates and values.

        Note: the dataset must have 'x', 'y', and 'time' coordinates.
        """
        dtype = dataset.z.dtype.type
        attrs = cls()
        attrs.actual_range = (dtype(dataset.z.min()), dtype(dataset.z.max()))
        if "scale_factor" and "add_offset" in dataset.z.attrs:
            attrs.scale_factor = dtype(dataset.z.attrs["scale_factor"])
            attrs.add_offset = dtype(dataset.z.attrs["add_offset"])
        return attrs

    @property
    def as_dict(self):
        return self.__dict__


class MetaAttrs(AbstractMetadataAttributes):
    """
    Dataclass for metadata attributes.
    """

    long_name: str = ""
    definition: str = ""
    values = None

    @classmethod
    def from_metadata_df(cls, metadata_df, indices, **kwargs):
        obj = cls()
        for k, v in kwargs.items():
            obj.__setattr__(k, v)
        obj.values = metadata_df.loc[indices][obj.__dict__["long_name"]].values
        if obj.values.dtype == "object":
            obj.values = np.array(
                [v.encode("utf-8") for v in obj.values.astype(np.str_)]
            )
        if hasattr(obj, "units"):
            if np.issubdtype(obj.values.dtype, np.datetime64):
                obj.__setattr__("units", "days since 1970-01-01 00:00:00")
                obj.__setattr__("calendar", "gregorian")
                obj.values = (
                    obj.values - np.datetime64("1970-01-01")
                ) / np.timedelta64(1, "D")
                obj.__setattr__(
                    "actual_range",
                    (
                        np.int32(obj.values.min()),
                        np.int32(obj.values.max()),
                    ),
                )
            else:
                obj.__setattr__(
                    "actual_range",
                    (np.nanmin(obj.values.min()), np.nanmax(obj.values.max())),
                )

        else:
            obj.__setattr__("units", "1")

        return obj

    @property
    def attrs_as_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "values"}


@dataclass
class NlhoGlobalAttributes:
    """
    Dataclass for NLHO survey metadata attributes.

    Includes defaults for most attributes or dynamically sets them based on the current
    time, user and computer. The date_modified attribute is set to the current time when
    any other attribute is changed.

    Geospatial attributes can be set using the from_dataarray class method, which takes
    a dataarray with 'x', 'y', and 'time' coordinates to determine attribute values and
    construct the NlhoGlobalAttributes object.
    """

    id: str = (
        f"hydrografie_survey_grids_release_{get_current_datetime(format='%Y%m%d')}"
    )
    naming_authority: str = "deltares.nl"
    Conventions: str = "CF-1.8"
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
    date_created: str = f"{get_current_datetime()}"
    date_modified: str = f"{get_current_datetime()}"
    timecoverage: str = ""
    time_coverage_units: str = "days since 1970-01-01 00:00:00"
    vertical_datum: str = "LAT"
    projectioncoverage_x: list = None
    projectioncoverage_y: list = None
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

    @classmethod
    def from_dataset(cls, dataset, timeaxis=True):
        """
        Create an NLHOAttributes object from an xarray.Dataset. This method sets the
        geospatial global attributes based on the dataset's coordinates and values.

        Note: the dataset must have 'x', 'y', and 'time' coordinates.
        """
        attrs = cls()

        dataset_crs = CRS(dataset.crs.attrs["projected_crs_name"])
        x_min_projected = float(dataset.x.min())
        x_max_projected = float(dataset.x.max())
        y_min_projected = float(dataset.y.min())
        y_max_projected = float(dataset.y.max())

        attrs.geospatial_lon_min, attrs.geospatial_lat_min = xy_to_ll(
            x_min_projected, y_min_projected, dataset_crs
        )
        attrs.geospatial_lon_max, attrs.geospatial_lat_max = xy_to_ll(
            x_max_projected, y_max_projected, dataset_crs
        )
        attrs.projectioncoverage_x = (x_min_projected, x_max_projected)
        attrs.projectioncoverage_y = (y_min_projected, y_max_projected)
        attrs.geospatialcoverage_eastwest = (
            attrs.geospatial_lon_min,
            attrs.geospatial_lon_max,
        )
        attrs.geospatialcoverage_northsouth = (
            attrs.geospatial_lat_min,
            attrs.geospatial_lat_max,
        )
        attrs.geospatial_vertical_min = float(dataset.z.min())
        attrs.geospatial_vertical_max = float(dataset.z.max())
        if timeaxis:
            attrs.timecoverage = (
                f"{dataset.time.min().values} - {dataset.time.max().values}"
            )
        return attrs

    @property
    def as_dict(self):
        return self.__dict__
