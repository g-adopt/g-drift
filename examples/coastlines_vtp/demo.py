# Reconstructing Coastlines as VTK Time-Series
# ==============================================
#
# This example demonstrates how to turn a GPlates plate-reconstruction
# model into a ParaView time-series (``.pvd`` + ``.vtp``) that can be
# overlaid on 3-D geodynamic model output. The conversion is handled by
# ``gdrift.CoastlineVTKFile``, which wraps
# `pygplates <https://www.gplates.org/docs/pygplates/>`_ for rotation,
# `mapbox_earcut <https://github.com/skogler/mapbox_earcut_python>`_ for
# in-plane polygon triangulation, and
# `pyvista <https://docs.pyvista.org/>`_ for mesh refinement and output.
#
# Background
# ----------
#
# Mantle convection simulations are usually visualised in a frame that
# rotates with the deep Earth, and their outputs live on a unit sphere
# (or a sphere scaled by ``R_earth``). Overlaying present-day coastlines
# is only meaningful at ``0`` Ma; for any earlier epoch the continents
# sit in different positions. GPlates solves this by combining a set of
# rotation files (``.rot``) with a coastlines feature collection
# (``.gpml``/``.gpmlz``) — given an age in millions of years, pygplates
# reconstructs where each polygon was at that time.
#
# ``CoastlineVTKFile`` drives this process and writes the output in a
# shape ParaView can animate:
#
# - one ``.vtp`` per age, containing the reconstructed polygons as
#   polylines on a sphere of user-chosen radius;
# - a top-level ``.pvd`` that collects them into a time-series, sorted
#   from oldest to youngest so ParaView's timeline runs forward in time.
#
# This example
# ------------
#
# We will:
#
# 1. Point ``CoastlineVTKFile`` at a rotation model and coastline file.
# 2. Call ``write_vtp(age)`` for a handful of ages.
# 3. Inspect the generated ``.pvd`` and ``.vtp`` files.
#
# Inputs
# ------
#
# This demo needs two GPlates files (rotation model + coastlines) for
# the Muller et al. 2019 v2 reconstruction. They are resolved in three
# steps: the ``GDRIFT_DEMO_ROTATION_FILE`` / ``GDRIFT_DEMO_COASTLINES_FILE``
# environment variables, a local developer copy at a well-known path,
# and finally an auto-download from the gadopt CDN into a cache dir.

# +
import os
import urllib.request
from pathlib import Path

CDN_BASE = (
    "https://gadopt.syd1.cdn.digitaloceanspaces.com/g-drift/"
    "test_data/coastlines_vtp/"
)
ROTATION_NAME = "Muller_etal_2019_CombinedRotations.rot"
COASTLINES_NAME = "Muller_etal_2019_Global_Coastlines.gpmlz"

LOCAL_GPLATES_ROOT = Path(
    "/Users/sghelichkhani/Workplace/gtrack/Zahirovic/"
    "Muller_etal_2019_v2_PlateMotionModel"
)
CACHE_DIR = Path.home() / "Workplace" / "g-drift" / "gdrift" / "data-gplates"


def _resolve(env_var, filename):
    env_path = os.environ.get(env_var)
    if env_path and Path(env_path).exists():
        return Path(env_path)
    local = LOCAL_GPLATES_ROOT / filename
    if local.exists():
        return local
    cached = CACHE_DIR / filename
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        url = CDN_BASE + filename
        print(f"Downloading {url} -> {cached}")
        with urllib.request.urlopen(url) as resp, open(cached, "wb") as out:
            out.write(resp.read())
    return cached


ROTATION_FILE = _resolve("GDRIFT_DEMO_ROTATION_FILE", ROTATION_NAME)
COASTLINES_FILE = _resolve("GDRIFT_DEMO_COASTLINES_FILE", COASTLINES_NAME)
print(f"Rotation model:  {ROTATION_FILE}")
print(f"Coastlines file: {COASTLINES_FILE}")
# -

# Building the VTK writer
# -----------------------
#
# The constructor takes the output ``.pvd`` path, the rotation model(s),
# the coastlines file, a minimum polygon length (in number of exterior
# points) to drop small-island noise, and an ``earth_radius`` that sets
# the sphere the polygons live on. We use ``earth_radius=1.0`` so the
# output matches a unit-sphere geodynamic mesh; pass ``R_earth`` if your
# simulation uses metres.

# +
import gdrift  # noqa: E402

output_pvd = Path("coastlines.pvd")

writer = gdrift.CoastlineVTKFile(
    filename=output_pvd,
    rotation_model=ROTATION_FILE,
    coastlines=COASTLINES_FILE,
    minimum_length_of_polygon=100,
    earth_radius=1.0,
)
print(f"Output directory: {writer.vtp_dir}")
# -

# Writing a time-series
# ---------------------
#
# Each call to ``write_vtp(age)`` reconstructs the coastlines at that
# age, wraps polygons across the antimeridian, filters short polygons,
# triangulates each closed piece with ``mapbox_earcut`` in (lon, lat),
# adaptively refines the mesh on the sphere so every triangle lies on a
# great-circle patch, re-orients normals outward, and saves the combined
# result as a ``.vtp``. The ``.pvd`` is rewritten after every call so
# you can open it in ParaView as soon as the first age is done.
#
# Here we sample the last 100 Myr every 20 Myr.

# +
ages_ma = [0, 20, 40, 60, 80, 100]

for age in ages_ma:
    writer.write_vtp(age)
    print(f"  wrote age={age} Ma")
# -

# Inspecting the output
# ---------------------
#
# The writer produces one ``.vtp`` per age inside ``coastlines/`` and a
# top-level ``coastlines.pvd`` that references them all.

# +
print("\nGenerated files:")
print(f"  {output_pvd}  ({output_pvd.stat().st_size} bytes)")
for vtp in sorted(writer.vtp_dir.glob("*.vtp")):
    print(f"  {vtp}  ({vtp.stat().st_size} bytes)")

print("\nFirst few lines of the .pvd collection:")
print("\n".join(output_pvd.read_text().splitlines()[:6]))
# -

# Using the output in ParaView
# ----------------------------
#
# Open ``coastlines.pvd`` in ParaView. It shows up as a time-varying
# triangulated surface dataset on a sphere of radius ``earth_radius``.
# Set the representation to "Surface" (or "Surface With Edges" to see the
# triangulation). Load your geodynamic output alongside it (on the same
# radius) and scrub the timeline to see continents move over the
# convecting mantle.
#
# A few practical notes:
#
# - ``write_vtp`` currently accepts any numeric age; the filenames use
#   ``{age:g}`` so ``0``, ``12.5`` and ``100`` all produce clean names.
# - Passing a list of rotation files is supported — pygplates merges
#   them into a single ``RotationModel``.
# - If you need coastlines on a sphere in metres, set
#   ``earth_radius=gdrift.constants.R_earth``.
