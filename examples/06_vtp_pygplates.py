from pathlib import Path
from gdrift import CoastlineVTKFile

data_path = (
    Path(__file__).resolve().parent /
    '../gdrift/data-3/Muller_etal_2019_PlateMotionModel_v2.0_Tectonics/'
)

# The partition_into_plates function requires a rotation model, since sometimes this would be
# necessary even at present day (for example to resolve topological polygons)
coastline_finame = data_path / \
    'StaticGeometries/Coastlines/Global_coastlines_2019_v1_low_res.shp'


rotation_finames = [
    data_path / 'Global_250-0Ma_Rotations_2019_v2.rot',
    data_path / 'Global_410-250Ma_Rotations_2019_v2.rot'
]
ccs = CoastlineVTKFile(
    filename="./coastlines.pvd",
    rotation_model=rotation_finames,
    coastlines=coastline_finame,
    minimum_length_of_polygon=10,
    earth_radius=2.21,
)

for age in [20, 15, 10, 5]:
    ccs.write_vtp(age=int(age))
