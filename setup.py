import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

pkgs = setuptools.find_packages()
print('found these packages:', pkgs)

pkg_name='ml_ephys'

setuptools.setup(
    name=pkg_name,
    version="0.2.14",
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
        'mountainlab_pytools',
        'deepdish',
        'scipy',
        'numpydoc',
        'h5py'
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
)
