from setuptools import setup, find_packages
from setuptools.command.install import install
import urllib.request
import os


class CustomInstall(install):
    def run(self):
        # Run the standard install process
        install.run(self)

        # Directory inside the package where files should be downloaded
        package_directory = self.install_lib  # Path to where the package is installed
        target_dir = os.path.join(package_directory, 'data')

        # Create the directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)

        # URLs of the files you want to download
        file_urls = [
            "https://zenodo.org/records/10971820/files/SLB_16_basalt.hdf5",
            "https://zenodo.org/records/10971820/files/SLB_16_pyrolite.hdf5",
        ]

        # Download and save the files
        for url in file_urls:
            filename = url.split('/')[-1]
            file_path = os.path.join(target_dir, filename)
            urllib.request.urlretrieve(url, file_path)
            print(f"Downloaded {filename} to {file_path}")


setup(
    name='gdrift',
    version='0.1',
    packages=find_packages(),
    description='Geodynamics Data Reformatting and Integration Facilitation Toolkit',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Sia Ghelichkhan',
    author_email='siavash.ghelichkhan@anu.edu.au',
    url='https://github.com/sghelichkhani/g-drift',
    install_requires=[
        'numpy',
        'scipy',
        'h5py',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    cmdclass={
        'install': CustomInstall,
    },
)
