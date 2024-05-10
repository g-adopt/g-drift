from pathlib import Path
from scipy import misc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from lib_anelasticity import load_solidus, dens_grav_pres_prof
from os.path import join
import gdrift

import h5py

# # Create a new HDF5 file
# with h5py.File('solidus_temperature_studies.hdf5', 'w') as f:
#     # Add file-level metadata
#     f.attrs['Title'] = 'Solidus Temperature Data Compilation'
#     f.attrs['Description'] = 'Aggregated solidus temperature measurements from multiple studies.'
#
#     # Create a group for a study
#     grp = f.create_group('Study1')
#     grp.attrs['Authors'] = 'Dr. A, Prof. B'
#     grp.attrs['Study Dates'] = '2021'
#     grp.attrs['Publication Reference'] = 'DOI or URL'
#
#     # Add dataset and metadata specific to the dataset
#     dset = grp.create_dataset('Temperature Data', (100,), dtype='f')
#     dset.attrs['Units'] = 'Celsius'
#     dset.attrs['Measurement Technique'] = 'Technique Description'


# hirschmann_solidus


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


#
moth_path = Path.home() / 'Workplace/PYTHBOX/DATA/SOLIDUS/'

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


dpth_file, pres_file = np.loadtxt(
    Path.home() / 'Workplace/PYTHBOX/DATA/min_table_z_P.txt', unpack=True)
dpth_file = dpth_file[pres_file > 0]
pres_file = pres_file[pres_file > 0]
T_solidus_file = load_solidus(
    dpths=dpth_file, mode='const_grad', solgrad=1.1, T50=1326.0)+273

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

mantle_dpths = np.linspace(10, 2790, 1401)
mantle_rads = 6371 - mantle_dpths
dens, grav, mantle_pres = dens_grav_pres_prof(mantle_rads*1e3)

mantle_solidus = interp1d(own_pres[own_pres.argsort()], own_solidus[own_pres.argsort()],
                          kind='linear', fill_value='extrapolate')(mantle_pres*1e-9)


# plt.close(1)
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8), num=1)
# for i in range(len(solidus_files)):
#     pres, temperature = np.loadtxt(
#         join(moth_path, solidus_files[i]), delimiter=',', unpack=True)
#     ax.plot(pres[pres.argsort()], temperature[pres.argsort()],
#             label=solidus_legends[i])
# ax.plot(pres_file, T_solidus_file, label='Richards Hoggard')
# ax.plot(hirsch_pres, hirsch_solid, label='Hirschmann 2000')
# ax.plot(katz_pres, katz_soli, label='Katz et al 2010')
# ax.plot(mantle_pres*1e-9, mantle_solidus, '--',
#         linewidth=1, color='k', label='Sia G')
# ax.grid()
# ax.set_xlabel('Pressure [GPa]')
# ax.set_ylabel('Temperature [K]')
# plt.legend()
# # fig.show()
# fig.savefig('../../../../IMG/Solidus_Temp_comparison.png', dpi=200)
#
# fi = open(file='../../../../DATA/SOLIDUS/SiaG_solidus.csv', mode='w')
# fi.write('# This solidus is used by Siavash Ghelich\n# constis of Hirschmann 2000' +
#          'in the upper mantle\n# and Andrault et al 2011 in the lower mantle.\n' +
#          '# Depth[m], Pressure[Pa], Solidus[K]\n')
# for i in range(len(mantle_solidus)):
#     fi.write('%10.2f, %10.4e, %7.3f\n' %
#              (mantle_dpths[i]*1e3, mantle_pres[i], mantle_solidus[i]))
# fi.close()
#
