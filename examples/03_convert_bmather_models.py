""" Here we convert all the models from all our source files
    to xyz values. The idea behind this is that the models
    are here provided in longitude and latitude.
    But then this is a bad coordinate system as it does not
    appreciate
"""

from gdrift import (
    fibonacci_sphere, cartesian_to_geodetic,
    R_earth, create_dataset_file
)
import warnings
import netCDF4 as nc
from pathlib import Path
from scipy.spatial import cKDTree
from gdrift.io import load_dataset
import numpy as np
# warnings.simplefilter('error', UserWarning)


def __main__():
    # Where to find the files
    mother_path = Path(
        "/Users/sghelichkhani/Workplace/seismic-tomography-models")

    global all_model_names
    all_model_names = []

    all_model_names = set(
        [filename.name.split("_")[0] for
         filename in mother_path.glob("*.nc")])

    for model_name in all_model_names:
        if "REVEAL.nc" == str(model_name):
            continue
        # Converting tomography model to our format
        convert_mather_file_to_mine(model_name=mother_path / model_name)


def convert_mather_file_to_mine(model_name):
    # Number of equidistantial points on a unit sphere
    num_samples = 181 * 361
    num_neighbours = 10

    # generating an equidistantial
    equidists_coords = fibonacci_sphere(n=num_samples)

    coords = {}
    meta_data = {}
    all_interpolation_results = {}

    for file_name in model_name.parents[0].glob(model_name.name + "*.nc"):
        data_type = file_name.name.split("_")[-1].replace(".nc", "")
        coords[data_type], all_interpolation_results[data_type], meta_data[data_type] = (
            interpolate_single_file_from_mather(
                filename=file_name,
                coords_on_sphere=equidists_coords,
                num_neighbours=num_neighbours)
        )

    if len(meta_data.keys()) > 1:
        are_arrays_equal(*[coords[key]
                         for key in coords.keys()])

    meta_data = next(iter(meta_data.values()))

    # Chaning meta_data, so people know what we have done
    meta_data[
        "comment"] = meta_data.get("comment", "") + (
            "\n This has been remapped to equidistantial points (181x361"
            "of fibanocci points) on a sphere by Sia Ghelichkhan."
            " (siavash.ghelichkhan@anu.edu.au)\n"
    )
    data_to_write = {}
    data_to_write["coordinates"] = next(iter(coords.values()))

    # setting a scaling factor to turn things to m/s

    # Adding all the data to the dataset
    for key in all_interpolation_results.keys():
        #  Scaling factor for vsh and vsv to m/s
        scaling_factor = 1e3 if key in ["vsh", "vsv"] else 1.0
        data_to_write[key] = all_interpolation_results[key][key] * scaling_factor

    # Printing what dataset we have
    print(f"{model_name.name} has {", ".join([key for key in data_to_write.keys()])}")

    # Write out datasaet
    create_dataset_file(
        f"3d_seismic_{model_name.name}.h5",
        data_info=data_to_write,
        metadata=meta_data
    )


def get_global_attributes(nc_dataset):
    attrs = {}
    for attr_name in nc_dataset.ncattrs():
        attrs[attr_name] = getattr(nc_dataset, attr_name)
    return attrs


def get_dimensions(nc_dataset):
    dims = {}
    for dim_name, dim in nc_dataset.dimensions.items():
        dims[dim_name] = len(dim)
    return dims


def get_variables(nc_dataset):
    vars_dict = {}
    for var_name, var in nc_dataset.variables.items():
        vars_dict[var_name] = {
            'data': var[:],  # Extract the data
            'dimensions': var.dimensions,  # Dimensions of the variable
            'attributes': {attr_name: getattr(var, attr_name) for attr_name in var.ncattrs()}
        }
    return vars_dict


def interpolate_single_file_from_mather(
        filename, coords_on_sphere, num_neighbours=10):

    # Converting equidistantial points to lon lat
    equidists_lats, equidists_lons, _ = cartesian_to_geodetic(
        *coords_on_sphere.T)

    # Opening the dataset file
    dataset = nc.Dataset(filename.resolve(), "r", maskandscale=False)
    dataset.set_auto_maskandscale(False)

    # Metadata should be stored seperately
    variables = {}

    # Getting all the variables
    for variable_name in dataset.variables.keys():
        variables[variable_name] = dataset.variables[variable_name][:]

    # adjusting for longitudes if they are larger that 180
    variables["longitude"][variables["longitude"] > 180.0] += -360.0

    meta_data = get_global_attributes(dataset)

    # closing the file
    dataset.close()

    # Depending on the model we might have different number of layers
    num_layers = variables["depth"].shape[0]

    # generate a grid based on longitudes and latitudes of the model
    lons_x, lats_x = np.meshgrid(
        variables["longitude"], variables["latitude"], indexing="xy")

    # A KDTree to find the nearest neighbours
    tree = cKDTree(np.column_stack((lons_x.flatten(), lats_x.flatten())))

    # shortes distances and indices
    dists, indices = tree.query(np.column_stack(
        (equidists_lons, equidists_lats)), k=num_neighbours)

    # for close_points we do not do interpolation
    close_points = dists[:, 0] < 1e-6

    # raise a warning if we do not have a distance within 1.0 degrees
    far_points = dists[:, 0] > 5.0
    if any(far_points):
        warnings.warn(
            "KDTree has found distances larger than 1.0.", UserWarning)

    # Suppress warnings for division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        weighted_interpolation = np.asarray(
            np.einsum(
                "jk,ijk->ij",
                1 / dists,
                np.asarray(
                    [variables["v"][i, :, :].flatten()[indices]
                     for i in range(num_layers)])) / np.sum(1 / dists, axis=1)
        )

    # points too close are not interpolated
    weighted_interpolation[:, close_points] = (
        [variables["v"][i, :, :].flatten()[indices[close_points, 0]]
         for i in range(num_layers)]
    )
    # points too far are not assigned
    weighted_interpolation[:, far_points] = np.nan

    # flatten all the results
    weighted_interpolation = weighted_interpolation.flatten()

    # set up coordinates
    all_coordinates = np.asarray(
        [coords_on_sphere * (R_earth - variables["depth"][i] * 1e3)
         for i in range(num_layers)]).reshape(num_layers * coords_on_sphere.shape[0], 3)

    data_to_write = {
        "coordinates": all_coordinates,
        str(filename).split("_")[-1].replace(".nc", ""): weighted_interpolation
    }

    return all_coordinates, data_to_write, meta_data


def are_arrays_equal(*arrays):
    # Compare the first array with all others
    first_array = arrays[0]
    return all(np.array_equal(first_array, array) for array in arrays[1:])


def convert_reveal():
    # path to the reveal model
    reveal_path = Path(
        "/Users/sghelichkhani/Workplace/seismic-tomography-models/REVEAL.nc")

    # Number of equidistantial points on a unit sphere
    num_samples = 181 * 361
    num_neighbours = 10

    # generating an equidistantial
    equidists_coords = fibonacci_sphere(n=num_samples)

    coords, data_to_write, meta_data = interpolate_reveal(
        filename=reveal_path,
        coords_on_sphere=equidists_coords,
        num_neighbours=num_neighbours)

    for key in data_to_write.keys():
        if 'v' in key:
            data_to_write[key] *= 1e3

    # Chaning meta_data, so people know what we have done
    meta_data[
        "comment"] = meta_data.get("comment", "") + (
            "\n This has been remapped to equidistantial points (181x361"
            "of fibanocci points) on a sphere by Sia Ghelichkhan."
            " (siavash.ghelichkhan@anu.edu.au)\n"
    )

    # Write out datasaet
    create_dataset_file(
        "3d_seismic_REVEAL.h5",
        data_info=data_to_write,
        metadata=meta_data
    )


def interpolate_reveal(
        filename, coords_on_sphere, num_neighbours=10):

    # Converting equidistantial points to lon lat
    equidists_lats, equidists_lons, _ = cartesian_to_geodetic(
        *coords_on_sphere.T)

    # Opening the dataset file
    dataset = nc.Dataset(filename.resolve(), "r")

    # Metadata should be stored seperately
    variables = {}

    for variable_name in dataset.variables.keys():
        variables[variable_name] = dataset.variables[variable_name][:]

    # adjusting for longitudes if they are larger that 180
    variables["longitude"][variables["longitude"] > 180.0] += -360.0

    meta_data = get_global_attributes(dataset)

    # closing the file
    dataset.close()

    # Depending on the model we might have different number of layers
    num_layers = variables["depth"].shape[0]

    # generate a grid based on longitudes and latitudes of the model
    lons_x, lats_x = np.meshgrid(
        variables["longitude"], variables["latitude"], indexing="xy")

    # A KDTree to find the nearest neighbours
    tree = cKDTree(np.column_stack((lons_x.flatten(), lats_x.flatten())))

    # shortes distances and indices
    dists, indices = tree.query(np.column_stack(
        (equidists_lons, equidists_lats)), k=num_neighbours)

    # for close_points we do not do interpolation
    close_points = dists[:, 0] < 1e-6

    # raise a warning if we do not have a distance within 1.0 degrees
    far_points = dists[:, 0] > 5.0
    if any(far_points):
        warnings.warn(
            "KDTree has found distances larger than 1.0.", UserWarning)

    weighted_interpolation = {}

    # Suppress warnings for division by zero
    for key in list(variables.keys())[3:]:
        with np.errstate(divide='ignore', invalid='ignore'):
            weighted_interpolation[key] = np.asarray(
                np.einsum(
                    "jk,ijk->ij",
                    1 / dists,
                    np.asarray(
                        [variables[key][i, :, :].flatten()[indices]
                         for i in range(num_layers)])) / np.sum(1 / dists, axis=1)
            )

        # points too close are not interpolated
        weighted_interpolation[key][:, close_points] = (
            [variables[key][i, :, :].flatten()[indices[close_points, 0]]
             for i in range(num_layers)]
        )
        # points too far are not assigned
        weighted_interpolation[key][:, far_points] = np.nan

        # flatten all the results
        weighted_interpolation[key] = weighted_interpolation[key].flatten()

    # set up coordinates
    all_coordinates = np.asarray(
        [coords_on_sphere * (R_earth - variables["depth"][i] * 1e3)
         for i in range(num_layers)]).reshape(num_layers * coords_on_sphere.shape[0], 3)

    weighted_interpolation["coordinates"] = all_coordinates

    return all_coordinates, weighted_interpolation, meta_data


if __name__ == "__main__":
    convert_reveal()
    # __main__()
