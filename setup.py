from setuptools import setup, find_packages

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
    install_requires=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

