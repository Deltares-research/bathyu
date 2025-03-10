from unittest.mock import patch

import numpy as np
import xarray as xr

from bathyu.io.opendap import list_opendap_files, nlho_tiles_from_bbox


def test_list_opendap_files_success():
    base_url = "https://example.com/opendap"
    html_content = """
    <html>
        <body>
            <a href="file1.nc">file1.nc</a>
            <a href="file2.nc">file2.nc</a>
            <a href="file3.nc">file3.nc</a>
        </body>
    </html>
    """

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = html_content

        files = list_opendap_files(base_url)
        assert files == ["file1.nc", "file2.nc", "file3.nc"]


def test_list_opendap_files_failure():
    base_url = "https://example.com/opendap"

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 404

        files = list_opendap_files(base_url)
        assert files == []


def test_list_opendap_files_no_files():
    base_url = "https://example.com/opendap"
    html_content = """
    <html>
        <body>
            <a href="?query">query</a>
            <a href="/path">path</a>
        </body>
    </html>
    """

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = html_content

        files = list_opendap_files(base_url)
        assert files == []


def test_nlho_tiles_from_bbox_success():
    base_url = "https://example.com/opendap"
    html_content = """
    <html>
        <body>
            <a href="x405000y5660000.nc">x405000y5660000.nc</a>
            <a href="x410000y5665000.nc">x410000y5660000.nc</a>
        </body>
    </html>
    """
    dataset_mock = xr.Dataset({"data": (["x", "y"], np.random.rand(10, 10))})

    with (
        patch("requests.get") as mock_get,
        patch("xarray.open_dataset") as mock_open_dataset,
    ):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = html_content
        mock_open_dataset.return_value = dataset_mock

        # Situation 1: Bounding box is within a single tile
        datasets = nlho_tiles_from_bbox(405000, 5660000, 408000, 5665000, base_url)
        assert len(datasets) == 1
        assert all(isinstance(ds, xr.Dataset) for ds in datasets)

        # Situation 2: Bounding box spans both tiles
        datasets = nlho_tiles_from_bbox(405000, 5660000, 412000, 5665000, base_url)
        assert len(datasets) == 2
        assert all(isinstance(ds, xr.Dataset) for ds in datasets)


def test_nlho_tiles_from_bbox_no_files():
    base_url = "https://example.com/opendap"
    html_content = """
    <html>
        <body>
            <a href="?query">query</a>
            <a href="/path">path</a>
        </body>
    </html>
    """

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = html_content

        datasets = nlho_tiles_from_bbox(405000, 5660000, 410000, 5665000, base_url)
        assert len(datasets) == 0


def test_nlho_tiles_from_bbox_partial_files():
    base_url = "https://example.com/opendap"
    html_content = """
    <html>
        <body>
            <a href="x405000y5660000.nc">x405000y5660000.nc</a>
            <a href="?query">query</a>
        </body>
    </html>
    """
    dataset_mock = xr.Dataset({"data": (["x", "y"], np.random.rand(10, 10))})

    with (
        patch("requests.get") as mock_get,
        patch("xarray.open_dataset") as mock_open_dataset,
    ):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = html_content
        mock_open_dataset.return_value = dataset_mock

        datasets = nlho_tiles_from_bbox(405000, 5660000, 410000, 5665000, base_url)
        assert len(datasets) == 1
        assert isinstance(datasets[0], xr.Dataset)
