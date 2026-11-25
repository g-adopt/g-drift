"""Tests for WGS84 positions and local frames."""

import numpy as np

from gdrift import ecef_to_enu_matrix, ecef_to_ned_matrix, wgs84_to_ecef


def test_wgs84_poles_and_equator():
    equatorial_radius = 6378137.0
    polar_radius = 6356752.314245179
    np.testing.assert_allclose(
        wgs84_to_ecef([0, 0, 90, -90], [0, 90, 0, 0]),
        [[equatorial_radius, 0, 0], [0, equatorial_radius, 0], [0, 0, polar_radius], [0, 0, -polar_radius]],
        atol=1e-9, rtol=0,
    )


def test_wgs84_height_follows_ellipsoid_normal():
    latitude = np.array([[-65.25], [45.0]])
    longitude = np.array([-61.65, 0, 90])
    heights_m = np.array([[-25.0, 0.0, 120.0]])
    surface = wgs84_to_ecef(latitude, longitude)
    positions = wgs84_to_ecef(latitude, longitude, heights_m)
    assert positions.shape == (2, 3, 3)
    up = ecef_to_enu_matrix(latitude, longitude)[..., 2, :]
    np.testing.assert_allclose(positions - surface, heights_m[..., None] * up, atol=1e-9, rtol=0)


def test_local_frames_at_equator():
    enu = ecef_to_enu_matrix(0, 90)
    ned = ecef_to_ned_matrix(0, 90)
    np.testing.assert_allclose(enu, [[-1, 0, 0], [0, 0, 1], [0, 1, 0]], atol=1e-15, rtol=0)
    np.testing.assert_allclose(ned, [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], atol=1e-15, rtol=0)


def test_broadcasted_local_frames_are_right_handed():
    latitude = np.array([[-90], [-65], [0], [47], [90]])
    longitude = np.array([-160, -60, 0, 120])
    for rotation in (ecef_to_enu_matrix, ecef_to_ned_matrix):
        frame = rotation(latitude, longitude)
        assert frame.shape == (5, 4, 3, 3)
        np.testing.assert_allclose(
            frame @ np.swapaxes(frame, -1, -2), np.broadcast_to(np.eye(3), frame.shape), atol=1e-15, rtol=0,
        )
        np.testing.assert_allclose(np.linalg.det(frame), 1, atol=1e-15, rtol=0)
