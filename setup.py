import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml_ephys",
    version="0.0.1",
    author="Jeremy Magland",
    author_email="",
    description="ephys tools for MountainLab",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/magland/ml_ephys",
    packages=setuptools.find_packages(),
    package_data={
        # Include all processor files
        '': ['*.mp']
    },
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
)
