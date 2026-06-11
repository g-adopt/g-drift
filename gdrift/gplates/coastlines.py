"""GPlates plate reconstruction integration for ParaView visualization.

This module provides optional integration with PyGPlates for reconstructing
coastline geometries through geological time and exporting them to VTK format
for visualization in ParaView. Each coastline polygon is triangulated on the
sphere and written as a filled surface, so continents can be overlaid on 3-D
geodynamic model output and animated through time.

**Note**: This module requires optional dependencies (pygplates, pyvista,
mapbox_earcut) that are not installed by default. It is primarily intended
for advanced users who need to visualize geodynamic models in 3D alongside
tectonic reconstructions.

Key Classes
-----------
CoastlineVTKFile : Reconstruct and export coastlines as time-series VTK

Key Methods
-----------
CoastlineVTKFile.reconstructed_coastlines : Get coastlines at specific age
CoastlineVTKFile.write_vtp : Write a single age as a .vtp (and update the .pvd)

Examples
--------
>>> import gdrift
>>> # Requires pygplates, pyvista and mapbox_earcut installed
>>> coastlines = gdrift.CoastlineVTKFile(
...     filename="coastlines.pvd",
...     rotation_model="rotation_files.rot",
...     coastlines="coastline_polygons.gpml",
...     earth_radius=1.0)  # normalized radius
>>> for age in range(0, 110, 10):
...     coastlines.write_vtp(age)
>>> # Open coastlines.pvd in ParaView to visualize time evolution

Notes
-----
- Optional dependencies: `pip install pygplates pyvista mapbox_earcut`
- Input files are GPlates format (.rot, .gpml, .gpmlz)
- Output is ParaView time-series (.pvd + directory of .vtp files)
- Coastlines are filtered by polygon length to remove small artifacts
- Coordinate system can be normalized to unit sphere for consistency with
  geodynamic models
- Polygons that enclose a geographic pole (notably Antarctica) are not
  rendered correctly: earcut triangulates in (lon, lat) and does not
  close the ring across the pole, so such polygons appear as strips with
  a polar hole. Fixing this requires injecting the pole as an extra
  vertex or meshing in a polar projection.
"""

from pathlib import Path
import numpy as np
import pyvista as pv
import pygplates
import mapbox_earcut


class CoastlineVTKFile(object):
    """Reconstruct present-day coastlines to a given age and export them as
    sphere-conforming filled surfaces for ParaView.

    Each ``write_vtp(age)`` call reconstructs every coastline feature via
    pygplates, splits polygons that cross the antimeridian with
    ``DateLineWrapper``, triangulates each sub-polygon in (lon, lat) using
    ``mapbox_earcut``, refines the mesh adaptively on the sphere so long
    thin "ear" triangles are broken into uniform patches, re-orients the
    normals outward, and saves the combined mesh as a ``.vtp`` inside the
    sibling directory of the ``.pvd`` collection.
    """

    # Max triangle edge length expressed in units of ``earth_radius``;
    # 0.01 corresponds to ~0.57 degrees on the sphere (~63 km on Earth).
    _MAX_EDGE_LEN = 0.01
    # Number of adaptive refinement passes. Each pass splits remaining
    # over-long edges and re-projects new vertices onto the sphere.
    _ADAPTIVE_PASSES = 4

    def __init__(
            self,
            filename,
            rotation_model,
            coastlines,
            minimum_length_of_polygon=100,
            earth_radius=1.0):

        # filename to be written out
        self.filename = Path(str(filename)).absolute().resolve()
        if self.filename.suffix != ".pvd":
            raise ValueError(f"{self.filename} should end in .pvd.")

        self.vtp_dir = self.filename.with_suffix("")
        if not self.vtp_dir.exists():
            self.vtp_dir.mkdir()

        # if rotation model is not a list make it a list
        if not isinstance(rotation_model, list):
            rotation_model = [rotation_model]

        # record rotation models
        self.rotation_model = pygplates.RotationModel(
            [Path(f).as_posix() for f in rotation_model])

        # coastline is a single shape file
        self.coastlines = pygplates.FeatureCollection(
            Path(coastlines).as_posix())

        # keeping record of what we are writing out for the pvd file
        self._vtp_records = []

        # do not write out polygons less complicated than this
        self.minimum_length_of_polygon = minimum_length_of_polygon

        # what should be radius of the Earth for sake of visualisation
        self.earth_radius = earth_radius

    def reconstructed_coastlines(self, age):
        """
        rotate back coastlines using pygplates
        age is in Ma
        """
        reconstructed_coastlines = []
        pygplates.reconstruct(self.coastlines,
                              self.rotation_model,
                              reconstructed_coastlines,
                              age)
        return reconstructed_coastlines

    def _wrap_coastlines(self, input_coastlines):
        # pygplates.DateLineWrapper is a wrapper around polygons
        # to deal with anamolous polygons where points far away
        # are connected to each other
        date_line_wrapper = pygplates.DateLineWrapper(0.0)

        # wrapping reconstructed coastline geometries to handle weird
        # line connections (see above for DateLineWrapper)
        return [
            date_line_wrapper.wrap(polygon.get_reconstructed_geometry())
            for polygon in input_coastlines
        ]

    def _write_pvd(self):

        lines = ['<?xml version="1.0"?>',
                 '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
                 '  <Collection>']
        for idx, (time, file) in enumerate(sorted(self._vtp_records, key=lambda x: x[0], reverse=True)):
            lines.append(f'    <DataSet timestep="{idx}" group="" part="0" file="{file.relative_to(self.filename.parent)}"/>')
        lines.append('  </Collection>')
        lines.append('</VTKFile>')
        self.filename.write_text('\n'.join(lines))

    def _project_to_sphere(self, points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return self.earth_radius * points / norms

    def _latlon_to_xyz(self, lats_deg, lons_deg):
        lat = np.radians(lats_deg)
        lon = np.radians(lons_deg)
        return self.earth_radius * np.stack([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ], axis=-1)

    def _orient_outward(self, mesh):
        # Flip any triangle whose normal points toward Earth's centre so the
        # filled continent renders from outside the globe regardless of the
        # source polygon's winding.
        mesh = mesh.compute_normals(
            cell_normals=True, point_normals=False,
            auto_orient_normals=False, consistent_normals=False,
            flip_normals=False, inplace=False,
        )
        centroids = mesh.cell_centers().points
        outward = np.einsum("ij,ij->i", centroids, mesh["Normals"]) > 0

        faces = mesh.faces.reshape(-1, 4).copy()
        inward_idx = np.where(~outward)[0]
        faces[inward_idx, 1:] = faces[inward_idx, 1:][:, ::-1]
        return pv.PolyData(mesh.points, faces)

    def _surface_from_latlon(self, latlon):
        # Triangulate the closed polygon directly in (lon, lat) with
        # earcut, which handles concavity correctly without a best-fit
        # plane projection. Then adaptively refine on the sphere so long
        # thin ear triangles become uniform patches.
        lats = latlon[:, 0]
        lons = latlon[:, 1]
        n = len(lats)
        if n < 3:
            return None

        lonlat = np.column_stack([lons, lats]).astype(np.float64)
        tri_flat = mapbox_earcut.triangulate_float64(lonlat, np.array([n]))
        tri_indices = np.asarray(tri_flat, dtype=np.int64).reshape(-1, 3)
        if len(tri_indices) == 0:
            return None

        points_xyz = self._latlon_to_xyz(lats, lons)
        faces = np.column_stack([
            np.full(len(tri_indices), 3, dtype=np.int64),
            tri_indices,
        ]).ravel()
        mesh = pv.PolyData(points_xyz, faces)

        max_edge = self._MAX_EDGE_LEN * self.earth_radius
        for _ in range(self._ADAPTIVE_PASSES):
            mesh = mesh.subdivide_adaptive(max_edge_len=max_edge)
            mesh.points = self._project_to_sphere(mesh.points)

        if mesh.n_cells == 0:
            return None
        return self._orient_outward(mesh)

    def write_vtp(self, age):
        # reconstructed coastlines using pygplates
        reconstructed_coastlines = self.reconstructed_coastlines(age)

        # DateLineWrapper splits antimeridian-crossing coastlines into
        # multiple closed pieces; each piece is triangulated on its own so
        # the filled surface never spuriously bridges disjoint parts.
        surfaces = []
        for wrapped in self._wrap_coastlines(reconstructed_coastlines):
            for sub_polygon in wrapped:
                latlon = np.asarray(
                    [p.to_lat_lon() for p in sub_polygon.get_exterior_points()]
                )
                if len(latlon) < self.minimum_length_of_polygon:
                    continue
                surf = self._surface_from_latlon(latlon)
                if surf is not None:
                    surfaces.append(surf)

        vtp_filename = self.vtp_dir / f"{self.filename.stem}_{age:g}.vtp"
        if surfaces:
            combined = surfaces[0] if len(surfaces) == 1 else np.sum(surfaces)
            combined.save(vtp_filename)
            self._vtp_records.append((age, vtp_filename))
            self._write_pvd()
