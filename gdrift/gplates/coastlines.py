from pathlib import Path
import numpy as np
import pyvista as pv
import pygplates


class CoastlineVTKFile(object):
    """
    is a class for visualising coastlines using vtk
    It rotates back present-day coastlines using pygplates

    methods:
        reconstructed_coastlines(age: Ma): returns a list of coastlines
             at age

    """

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

    def _record_call_time(method):
        def wrapper(self, age, *args, **kwargs):
            ret = method(self, age, *args, **kwargs)
            vtp_filename = (
                self.vtp_dir /
                f"{self.filename.stem}_{age:1d}.vtp"
            )
            self._vtp_records.append((age, vtp_filename))
            self._write_pvd()
            return ret
        return wrapper

    def _write_pvd(self):

        lines = ['<?xml version="1.0"?>',
                 '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
                 '  <Collection>']
        for idx, (time, file) in enumerate(sorted(self._vtp_records, key=lambda x: x[0], reverse=True)):
            lines.append(f'    <DataSet timestep="{idx}" group="" part="0" file="{file.relative_to(self.filename.parent)}"/>')
        lines.append('  </Collection>')
        lines.append('</VTKFile>')
        self.filename.write_text('\n'.join(lines))

    @_record_call_time
    def write_vtp(self, age):
        # reconstructed coastlines using pygplates
        reconstructed_coastlines = self.reconstructed_coastlines(age)

        # Get the exterior points
        polygons = [
            np.asarray(
                [
                    self.earth_radius * point.to_point_on_sphere().to_xyz_array().flatten()
                    for sub_polygon in polygon
                    for point in sub_polygon.get_exterior_points()
                ]
            )
            for polygon in self._wrap_coastlines(reconstructed_coastlines)
        ]

        # Make pyvista.PolyData out of polygons larger than
        # self.minimum_length_of_polygon
        polydata_list = [
            pv.PolyData(
                polygon, [len(polygon)] + list(range(len(polygon))))
            for polygon in polygons
            if len(polygon) > self.minimum_length_of_polygon
        ]

        # Sum all the polygons and write them out
        np.sum(polydata_list).save(self.vtp_dir / f"coastlines_{age}.vtp")
