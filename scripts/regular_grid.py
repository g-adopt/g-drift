"""Bilinear resampling of a regular latitude-longitude grid, shared by the conversion scripts.

scripts/convert_lloyd_viscosity.py and scripts/convert_glad_m35.py (and
through it scripts/convert_glad_m25.py) import this module. It is in
``scripts/`` and not in the gdrift package, because only the conversion of
source files into gdrift datasets uses it: a user of gdrift loads the
converted datasets and never resamples a source grid.

``scripts/`` is not a package. A script that imports this module puts its own
directory on ``sys.path`` first, so the import works both when the script is
run and when a test loads the script from its path.
"""
import numpy as np


def regular_grid_bilinear_weights(lat, lon, query_lat, query_lon):
    """Bilinear interpolation indices and weights on a regular latitude-longitude grid.

    Used by the conversion scripts to resample models that are published on
    a regular grid (for example EarthScope EMC files) onto the Fibonacci
    layers of the gdrift datasets. The same indices and weights
    apply to every depth layer, so they are computed once per model.

    The grid must contain every query point: no wrap-around in longitude is
    done, and a point outside the grid raises a ValueError. Query longitudes
    from `cartesian_to_geodetic` lie in [-180, 180], so a global grid must
    run from -180 to +180 with both ends included; a 0 to 360 grid must be
    converted first. A point on the last grid line uses the last cell with a
    fractional offset of 1, so no index runs past the end of the grid.

    Parameters
    ----------
    lat, lon : np.ndarray
        Increasing, evenly spaced grid axes in degrees.
    query_lat, query_lon : np.ndarray, shape (n,)
        Query points in degrees, inside the grid.

    Returns
    -------
    i0, j0 : np.ndarray of int, shape (n,)
        Indices of the lower-left grid node (in latitude and longitude) of the
        cell that contains each query point.
    weights : np.ndarray, shape (n, 4)
        Weights of the corners (i0, j0), (i0+1, j0), (i0, j0+1), (i0+1, j0+1),
        in that order. Each row sums to 1, and every weight is in [0, 1].

    Raises
    ------
    ValueError
        If a query point lies outside the grid.

    Examples
    --------
    >>> i0, j0, w = regular_grid_bilinear_weights(lat, lon, q_lat, q_lon)
    >>> corners = np.stack([field[..., i0, j0], field[..., i0 + 1, j0],
    ...                        field[..., i0, j0 + 1], field[..., i0 + 1, j0 + 1]], axis=-1)
    >>> values = np.sum(corners * w, axis=-1)
    """
    # Outside the grid the weights would extrapolate (negative weights), so
    # refuse. The tolerance absorbs the rounding of points on the grid edges.
    tol = 1e-9
    outside = ((query_lat < lat[0] - tol) | (query_lat > lat[-1] + tol)
               | (query_lon < lon[0] - tol) | (query_lon > lon[-1] + tol))
    if np.any(outside):
        raise ValueError(
            f"{int(np.sum(outside))} query points lie outside the grid "
            f"(latitude {lat[0]} to {lat[-1]}, longitude {lon[0]} to {lon[-1]})")

    dlat = lat[1] - lat[0]
    dlon = lon[1] - lon[0]
    # Fractional grid index of each query point
    fi = (query_lat - lat[0]) / dlat
    fj = (query_lon - lon[0]) / dlon
    # Lower corner, clipped so that the upper corner stays inside the grid
    i0 = np.clip(np.floor(fi).astype(int), 0, lat.size - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, lon.size - 2)
    # Offsets inside the cell, in [0, 1]
    ti = fi - i0
    tj = fj - j0
    weights = np.column_stack([
        (1 - ti) * (1 - tj),
        ti * (1 - tj),
        (1 - ti) * tj,
        ti * tj,
    ])
    return i0, j0, weights
