import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

pkgs = setuptools.find_packages()
print('found these packages:', pkgs)

setuptools.setup(
    name="ml_ephys",
    version="0.0.1",
    author="Jeremy Magland",
    author_email="",
    description="ephys tools for MountainLab",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/magland/ml_ephys",
    packages=pkgs,
    package_data={
        # Include all processor files
        '': ['*.mp']
    },
    install_requires=
    [
        'numpy',
        'mltools==0.1.2a',
        'deepdish',
        'scipy',
        'numpydoc',
        'h5py'
    ],
    dependency_links=['https://github.com/magland/mltools/tarball/master#egg=mltools-0.1.2a'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
)
