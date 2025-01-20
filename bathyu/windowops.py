import numba
import numpy as np
from numba import float32, int16
from numba.experimental import jitclass

__class_spec = [
    ("size", int16),
    ("array", float32[:, :]),
    ("row_center", int16),
    ("col_center", int16),
    ("row_min", int16),
    ("col_min", int16),
]


@jitclass(__class_spec)
class NumbaWindow:
    """
    Numba jitclass for moving window operations on 2D Numpy arrays. Input parameters
    must explicitly be according to specified datatypes below.

    Parameters
    ----------
    array : np.ndarray, (float32)
        Numpy array of shape (m, n).
    size : int16
        Size of the moving window, must be an odd number. A square window of
        size x size is created.

    """

    def __init__(self, array, size):
        self.array = array
        self.row_center = 0
        self.col_center = 0
        self.row_min = 0
        self.col_min = 0

        if size % 2 == 1:
            self.size = size
        else:
            raise ValueError("Only odd window size is allowed")

    @property
    def row_max(self):
        return self.array.shape[0]

    @property
    def col_max(self):
        return self.array.shape[1]

    @property
    def center_idxs(self):
        return self.row_center, self.col_center

    @property
    def window_radius(self):
        return int((self.size - 1) / 2)

    def get_mask_indices_from_center(self):
        """
        Determine the indices of the moving window to mask from the array based
        on the indices of the center of the window.

        Returns integers for the top, bottom, left and right of the window.

        """
        top_idx = self.row_center - self.window_radius
        bottom_idx = self.row_center + self.window_radius + 1

        left_idx = self.col_center - self.window_radius
        right_idx = self.col_center + self.window_radius + 1

        ## update indices for moving windows at edges that they don't fall out of range
        if top_idx < self.row_min:
            top_idx = self.row_min

        if bottom_idx > self.row_max:
            bottom_idx = self.row_max

        if left_idx < self.col_min:
            left_idx = self.col_min

        if right_idx > self.col_max:
            right_idx = self.col_max

        return top_idx, bottom_idx, left_idx, right_idx

    def window(self):
        """
        Return the array elements that are within the moving window.

        """
        top, bottom, left, right = self.get_mask_indices_from_center()
        window = self.array[top:bottom, left:right]
        return window

    def window_mean(self):
        return np.nanmean(self.window())

    def window_max(self):
        return np.nanmax(self.window())

    def window_min(self):
        return np.nanmin(self.window())

    def minimum_distance_min_max(self):
        window = self.window()

        if np.all(np.isnan(window)):
            dist = np.nan

        else:
            max_locs = np.argwhere(window == self.window_max())
            min_locs = np.argwhere(window == self.window_min())

            dists = numba_point_distance(min_locs, max_locs)

            dist = np.nanmin(dists)

        return dist


@numba.njit
def numba_window_average(array, window_size=3):
    """
    Numba optimized moving window average for 2D (m, n) Numpy arrays.

    Parameters
    ----------
    array : np.ndarray, (float32)
        Numpy array of shape (m, n).
    window_size : int16, optional
        Size of the moving window in number of elements. Only odd window size is
        allowed. The default is 3.

    Returns
    -------
    output : np.ndarray
        Numpy array of moving window average of shape (m, n).

    """
    nrows, ncols = array.shape
    output = np.full((nrows, ncols), np.nan)

    window = NumbaWindow(array, window_size)

    for row in range(nrows):
        window.row_center = row
        for col in range(ncols):
            window.col_center = col

            min_ = window.window_mean()
            output[row, col] = min_

    return output


@numba.njit
def numba_window_minimum(array, window_size=3):
    """
    Numba optimized moving window minimum for 2D (m, n) Numpy arrays.

    Parameters
    ----------
    array : np.ndarray, (float32)
        Numpy array of shape (m, n).
    window_size : int16, optional
        Size of the moving window in number of elements. Only odd window size is
        allowed. The default is 3.

    Returns
    -------
    output : np.ndarray
        Numpy array of moving window minimum of shape (m, n).

    """
    nrows, ncols = array.shape
    output = np.full((nrows, ncols), np.nan)

    window = NumbaWindow(array, window_size)

    for row in range(nrows):
        window.row_center = row
        for col in range(ncols):
            window.col_center = col

            min_ = window.window_min()
            output[row, col] = min_

    return output


@numba.njit
def numba_window_maximum(array, window_size=3):
    """
    Numba optimized moving window maximum for 2D (m, n) Numpy arrays.

    Parameters
    ----------
    array : np.ndarray, (float32)
        Numpy array of shape (m, n).
    window_size : int16, optional
        Size of the moving window in number of elements. Only odd window size is
        allowed. The default is 3.

    Returns
    -------
    output : np.ndarray
        Numpy array of moving window maximum of shape (m, n).

    """
    nrows, ncols = array.shape
    output = np.full((nrows, ncols), np.nan)

    window = NumbaWindow(array, window_size)

    for row in range(nrows):
        window.row_center = row
        for col in range(ncols):
            window.col_center = col

            max_ = window.window_max()
            output[row, col] = max_
    return output


@numba.njit
def numba_minimum_distance_min_max(array, window_size=3):
    """
    Numba optimized function to determine the minimum distance between min and
    max values within a moving window for 2D (m, n) Numpy arrays.

    Parameters
    ----------
    array : np.ndarray, (float32)
        Numpy array of shape (m, n).
    window_size : int16, optional
        Size of the moving window in number of elements. Only odd window size is
        allowed. The default is 3.

    Returns
    -------
    output : np.ndarray
        Numpy array of minimum distance in the moving window of shape (m, n).

    """
    nrows, ncols = array.shape
    output = np.full((nrows, ncols), np.nan)

    window = NumbaWindow(array, window_size)

    for row in range(nrows):
        window.row_center = row
        for col in range(ncols):
            window.col_center = col

            dist = window.minimum_distance_min_max()
            output[row, col] = dist

    return output


@numba.njit
def numba_point_distance(points_a, points_b):
    """
    Calculate the euclidean distance between all combinations of two sets of
    points. Both need to be in the same cartesian coordinate system.

    Parameters
    ----------
    points_a, points_b : ndarray
        Numpy array of size (q,2), where q is the amount of points.

    Returns
    -------
    distance : ndarray
        Numpy array of the distance between all point combinations. Specific
        combinations of points can be accessed according to their original index
        in the point arrays -> distance[point_a, point_b].

    """
    na = points_a.shape[0]
    nb = points_b.shape[0]

    distance = np.full((na, nb), np.nan)

    for i in range(na):
        xa, ya = points_a[i]
        for j in range(nb):
            xb, yb = points_b[j]

            dist = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
            distance[i, j] = dist

    return distance


@numba.njit
def numba_window_difference_between_minimum(array, window_size=3):
    """
    Numba optimized moving window for calculating the difference between the queried
    cell value and the minimum cell value within the window in 2D (m, n) Numpy arrays.

    Parameters
    ----------
    array : np.ndarray, (float32)
        Numpy array of shape (m, n).
    window_size : int16, optional
        Size of the moving window in number of elements. Only odd window size is
        allowed. The default is 3.

    Returns
    -------
    output : np.ndarray
        Numpy array of moving window maximum of shape (m, n).

    """
    nrows, ncols = array.shape
    output = np.full((nrows, ncols), np.nan)

    window = NumbaWindow(array, window_size)

    for row in range(nrows):
        window.row_center = row
        for col in range(ncols):
            window.col_center = col

            diff_min = window.window_min() - window.array[row, col]
            output[row, col] = diff_min
    return output


@numba.njit
def numba_window_difference_between_maximum(array, window_size=3):
    """
    Numba optimized moving window for calculating the difference between the queried
    cell value and the maximum cell value within the window in 2D (m, n) Numpy arrays.

    Parameters
    ----------
    array : np.ndarray, (float32)
        Numpy array of shape (m, n).
    window_size : int16, optional
        Size of the moving window in number of elements. Only odd window size is
        allowed. The default is 3.

    Returns
    -------
    output : np.ndarray
        Numpy array of moving window maximum of shape (m, n).

    """
    nrows, ncols = array.shape
    output = np.full((nrows, ncols), np.nan)

    window = NumbaWindow(array, window_size)

    for row in range(nrows):
        window.row_center = row
        for col in range(ncols):
            window.col_center = col

            diff_max = window.window_max() - window.array[row, col]
            output[row, col] = diff_max
    return output
