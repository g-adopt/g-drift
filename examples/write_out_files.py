import numpy as np
from os.path import join
from os import getenv
import gdrift
from scipy.interpolate import interp1d

metadata = {
    "Andrault et al 2011": {
        "Creator": 'Sia Ghelichkhan (siavash.ghelichkhan@anu.edu.au)',
        "Institution": 'Australian National University (RSES)',
        "Data_Source": "Solidus and liquidus profiles of chondritic mantle: Implication for melting of the Earth across its history",
        "title": "Preliminary reference Earth model",
        "author": "Andrault, Denis and Bolfan-Casanova, Nathalie and Nigro, Giacomo Lo and Bouhifd, Mohamed A and Garbarino, Gaston and Mezouar, Mohamed",
        "journal": "Earth and planetary science letters",
        "year": "2011",
        "publisher": "Elsevier",
        "DOI": "10.1016/j.epsl.2011.02.006",
        "Units": "SI: seismic velocities [m/s], density [kg/m^3], Radius [m]"
    },
    "Fiquet et al 2010": {
        "Creator": 'Sia Ghelichkhan (siavash.ghelichkhan@anu.edu.au)',
        "Institution": 'Australian National University (RSES)',
        "Data_Source": 'Contact Sia Ghelichkhan',
        "title": "Earth's mantle Solidus",
        "author": "Fiquet et al 2010",
        "journal": None,
        "year": "2010",
        "Units": "SI: seismic velocities [m/s], density [kg/m^3], Radius [m]"
    },
    "Nomura et al 2014": {
        "Creator": 'Sia Ghelichkhan (siavash.ghelichkhan@anu.edu.au)',
        "Institution": 'Australian National University (RSES)',
        "Data_Source": 'https://gfzpublic.gfz-potsdam.de/rest/items/item_43253/component/file_56084/content',
        "title": "Preliminary reference Earth model",
        "author": "Dziewonski, Adam M and Anderson, Don L",
        "journal": "Physics of the earth and planetary interiors",
        "year": "1981",
        "publisher": "Elsevier",
        "DOI": "https://doi.org/10.1016%2F0031-9201%2881%2990046-7",
        "Units": "SI: seismic velocities [m/s], density [kg/m^3], Radius [m]"
    },
    "Zerr et al 1988 (Upper Bound)": {
        "Creator": 'Sia Ghelichkhan (siavash.ghelichkhan@anu.edu.au)',
        "Institution": 'Australian National University (RSES)',
        "Data_Source": 'https://gfzpublic.gfz-potsdam.de/rest/items/item_43253/component/file_56084/content',
        "title": "Preliminary reference Earth model",
        "author": "Dziewonski, Adam M and Anderson, Don L",
        "journal": "Physics of the earth and planetary interiors",
        "year": "1981",
        "publisher": "Elsevier",
        "DOI": "https://doi.org/10.1016%2F0031-9201%2881%2990046-7",
        "Units": "SI: seismic velocities [m/s], density [kg/m^3], Radius [m]"
    },
}


def write_prem_out():
    fi_name = join(getenv('HOME'), 'Workplace/PYTHBOX/DATA/PREM_1s.csv')
    prem_arrays = np.loadtxt(
        fi_name, delimiter=',', unpack=True
    )

    with open(fi_name, mode="r") as fi:
        line = fi.readline()

    headers = line.split("represent ")[1].replace("]\n", "").split(",")

    # Example metadata
    metadata = {
        "Creator": 'Sia Ghelichkhan (siavash.ghelichkhan@anu.edu.au)',
        "Institution": 'Australian National University (RSES)',
        "Data_Source": 'https://gfzpublic.gfz-potsdam.de/rest/items/item_43253/component/file_56084/content',
        "title": "Preliminary reference Earth model",
        "author": "Dziewonski, Adam M and Anderson, Don L",
        "journal": "Physics of the earth and planetary interiors",
        "year": "1981",
        "publisher": "Elsevier",
        "DOI": "https://doi.org/10.1016%2F0031-9201%2881%2990046-7",
        "Units": "SI: seismic velocities [m/s], density [kg/m^3], Radius [m]"
    }
    print(headers)
    data_info = {}
    for index, head in enumerate(headers):
        data_info[head] = prem_arrays[index]
        if head in ["Vph", "Vsh", "Vsv", "Vpv", "density", "radius", "depth"]:
            data_info[head] *= 1000

    gdrift.create_dataset_file("1d_prem.h5", data_info, metadata)


def write_solidus_out():

    def hirschmann_solidus(pres):
        """
           This function generates a profile of the Earth's solidus
           based on model of Hirschmann 2000 for Peridotite
           See Hirschmann*.txt in Data
        """
        a = -5.904
        b = 139.44
        c = 1108.08
        # compute solidus in celsius
        T_solid = a*pres**2 + b*pres + c
        # convert to Kelvin
        T_solid += 273.0
        return T_solid

    prem = gdrift.PreRefEarthModel()

    #
    moth_path = '/Users/sghelichkhani/Workplace/PYTHBOX/DATA/SOLIDUS/'

    solidus_files = [
        'Andrault_et_al_2011_EPSL_solidus.csv',
        'Fiquet_et_al_2010_SCIENCE_solidus.csv',
        'Nomura_et_al_2014_SCIENCE_solidus.csv',
        'Zerr_et_al_1988_SCIENCE_solidus.csv'
    ]
    solidus_legends = [
        'Andrault et al 2011',
        'Fiquet et al 2010',
        'Nomura et al 2014',
        'Zerr et al 1988 (Upper Bound)'
    ]

    katz_dpth, katz_soli = np.loadtxt(
        join(moth_path, 'katz_solidi/zTF_1334_0.txt'),
        usecols=(0, 1), unpack=True)

    # prem.at_depth("density", katz_dpth)

    dpth_file, pres_file = np.loadtxt(
        join(moth_path, '../../DATA/min_table_z_P.txt'), unpack=True)
    dpth_file = dpth_file[pres_file > 0]
    pres_file = pres_file[pres_file > 0]
    # T_solidus_file = load_solidus(
    #     dpths=dpth_file, mode='const_grad', solgrad=1.1, T50=1326.0)+273

    # finding out katz_soli for dpth in min file
    katz_soli = interp1d(katz_dpth[katz_dpth.argsort()], katz_soli[katz_dpth.argsort()])(
        dpth_file[dpth_file < np.max(katz_dpth)])
    katz_pres = pres_file[dpth_file < np.max(katz_dpth)]

    # solidus by Hirschmann 2000
    hirsch_pres = np.linspace(0, 10, 100)
    hirsch_solid = hirschmann_solidus(hirsch_pres)

    # Build our own model
    own_pres, own_solidus = \
        np.loadtxt(join(moth_path, 'Andrault_et_al_2011_EPSL_solidus.csv'),
                   delimiter=',', unpack=True)

    own_pres = np.concatenate((own_pres, hirsch_pres), axis=0)
    own_solidus = np.concatenate((own_solidus, hirsch_solid), axis=0)
    # own_pres    = hirsch_pres;
    # own_solidus = hirsch_solid;

    mantle_rads = np.linspace(0, 6371e3, 1000)
    mantle_dpths = 6371e3 - mantle_rads
    mantle_mass = gdrift.compute_mass(
        mantle_rads, prem.at_depth("density", mantle_dpths))
    mantle_g = gdrift.compute_gravity(mantle_rads, mantle_mass)
    mantle_pres = gdrift.compute_pressure(
        mantle_rads, prem.at_depth("density", mantle_dpths), mantle_g)
    # _, _, mantle_pres_2 = dens_grav_pres_prof(mantle_rads*1e3)

    mantle_solidus = interp1d(own_pres[own_pres.argsort()], own_solidus[own_pres.argsort()],
                              kind='linear', fill_value='extrapolate')(mantle_pres*1e-9)

    pres_2_depths = interp1d(mantle_pres, mantle_dpths)

    for i, fi_name in enumerate(solidus_files):
        pres, temperature = np.loadtxt(
            join(moth_path, fi_name), delimiter=',', unpack=True)
        data_info = {}
        data_info["solidus temperature"] = temperature
        data_info["pressure"] = pres
        data_info["depth"] = pres_2_depths(pres*1e9)
        gdrift.create_dataset_file(
            f"1d_solidus_{fi_name.replace('_solidus.csv', '.h5')}",
            data_info,
            {"Authos": fi_name.replace(".csv", "")},
        )

    data_info = {}
    data_info["solidus temperature"] = mantle_solidus
    data_info["pressure"] = own_pres*1e9
    data_info["depth"] = pres_2_depths(own_pres)
    gdrift.create_dataset_file(
        "1d_solidus_Ghelichkhan_et_al_2021_GJI.h5",
        data_info,
        {"Authos": "Ghelichkhan et al (2021) GJI"},
    )


if __name__ == "__main__":
    # write_prem_out()
    write_solidus_out()
