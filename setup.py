import re
from setuptools import setup, Extension, find_packages


def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line and not line.startswith("#")]


one_deps = parse_requirements("requirements.txt")


earth_requires = [
    "pytz",
    "xarray",
    "zarr",
    "s3fs",
    "netcdf4",
    "cftime",
    "dask",
]


cfd_requires = [
    "shapely",
    "seaborn",
    "deepxde",
]

quantum_requires = [
    "openfermion",
    "pymatgen",
]


chemistry_requires = [
    "e3nn",
    "ase",
    "xtb",
    "rdkit",
    "matscipy",
    "python-hostlist",
    "configargparse",
    "lmdb",
    "orjson",
    "pymatgen",
]

biology_requires = [
    "rdkit",
    "matplotlib",
    "contextlib2",
    "ml-collections",
    "dm-tree",
    "dm-haiku",
    "diffrax",
    "biopandas",
    "biopython",
    "pyrsistent",
]

dev_requires = [
    "setuptools",
]


def parse_requirements_file(filename):
    """Parse a requirements file into a list of dependencies."""
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def resolve(requires, deps_dict):
    """Convert a list of dependencies to a set of requirements."""
    return [deps_dict[require] for require in requires]


extras = {}

install_requires = resolve(basic_requires, deps)
extras["earth"] = resolve(earth_requires, deps)
extras["bio"] = resolve(biology_requires, deps)
extras["cfd"] = resolve(cfd_requires, deps)
extras["chem"] = resolve(chemistry_requires, deps)
extras["quantum"] = resolve(quantum_requires, deps)
extras["dev"] = resolve(dev_requires, deps)


extras["all"] = one_deps

setup(
    name="onescience",
    version="0.1.2rc1",
    author="sugon-ai4s",
    author_email="ai4s@sugon.com",
    description="First release",
    long_description="OneScience is a scientific computing toolkit built on an advanced deep learning framework",
    url="https://github.com/hpccube/OneScience",
    package_dir={"": "src"},
    packages=find_packages("src"),
    # packages=find_packages(include=["*science*"]),
    extras_require=extras,
    include_package_data=True,
    install_requires=list(install_requires),
    python_requires=">=3.10.0",
    zip_safe=False,
)
