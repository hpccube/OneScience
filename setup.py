import re
from setuptools import setup, Extension, find_packages


one_requires = [
    'numpy>=1.25.0,<2.0.0',
    'xarray>=2023.1.0',
    'zarr>=2.14.2',
    's3fs==2023.9.0',
    'setuptools>=67.6.0',
    'pytz>=2023.3',
    'treelib>=1.2.5',
    'tqdm>=4.60.0',
 # 暂时不支持，需要去掉
    'timm>=0.9.12',
    'hydra-core>=1.2.0',
    'termcolor>=2.1.1',
    'wandb>=0.13.7',
    'mlflow>=2.1.1',
    'pytest>=6.0.0',
    'pyyaml>=6.0',
    'h5py>=3.7.0',
    'netcdf4>=1.6.3',
    'ruamel.yaml>=0.17.22',
    'scikit-learn>=1.2.2,<=1.6.0',
    'scikit-image>=0.24.0',
    'vtk>=9.2.6,<9.4.0',
    'pyvista>=0.40.1',
    'cftime>=1.6.2',
    'einops>=0.7.0',
    'shapely>=2.0.6',
    'onnx>=1.14.0',
    'pandas>=2.2.2',
    'omegaconf>=2.3.0',
    'dask>=2024.11.1',
    'mpi4py',
    'seaborn',
    'torchdata<=0.9.0',
    'openfermion==1.5.1',
    'pybind11',
    'torchmetrics',
    'e3nn==0.4.4',
    'deepxde',
    'ase',
    'rdkit',
    'xtb',
    'pymatgen',
    'torch-runstats>=0.2.0',
    'torch-ema>=0.3.0',
    'matscipy',
    'python-hostlist',
    'configargparse',
    'opt_einsum',
    'prettytable',
    'lmdb',
    'orjson',
    'matplotlib',
    'contextlib2',
    'ml-collections==0.1.1',
    'dm-tree==0.1.8',
    'dm-haiku==0.0.12',
    'diffrax==0.6.0',
    'biopandas==0.5.1',
    'biopython==1.84',
    'pyrsistent',
]

# {"numpy": "numpy>=1.25.0,<2.0.0",...}
deps_dict = {re.split(r'[=<>~!]', dep)[0]: dep for dep in one_requires}

basic_deps = [
    'numpy',
    'tqdm',
    'timm',
    'wandb',
    'hydra-core',
    'treelib',
    'hydra-core',
    'termcolor',
    'mlflow',
    'pytest',
    'pyyaml',
    'h5py',
    'ruamel.yaml',
    'scikit-learn',
    'scikit-image',
    'vtk',
    'pyvista',
    'einops',
    'onnx',
    'pandas',
    'omegaconf',
    'mpi4py',
    'torchdata',
    'pybind11',
    'torchmetrics',
    'torch-runstats',  # 性能分析
    'torch-ema',  #
    'opt_einsum',
    'prettytable',
    'matplotlib',
]


earth_deps = [
    'pytz',
    'xarray',
    'zarr',
    's3fs',
    'netcdf4',
    'cftime',
    'dask',

]


cfd_deps = [
    'shapely',
    'seaborn',
    'deepxde',
]

quantum_deps = [
    'openfermion',
    'pymatgen',
]


chemistry_deps = [
    'e3nn',
    'ase',
    'xtb',
    'rdkit',
    'matscipy',
    'python-hostlist',
    'configargparse',
    'lmdb',
    'orjson',
    'pymatgen',
]

biology_deps = [
    'rdkit',
    'matplotlib',
    'contextlib2',
    'ml-collections',
    'dm-tree',
    'dm-haiku',
    'diffrax',
    'biopandas',
    'biopython',
    'pyrsistent',
]

dev_deps = [
    'setuptools',
]


def parse_requirements_file(filename):
    """Parse a requirements file into a list of dependencies."""
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def deps_to_requires(deps):
    """Convert a list of dependencies to a set of requirements."""
    return [deps_dict[dep] for dep in deps if dep in deps_dict]


extras = {}

basic_requires = deps_to_requires(basic_deps)
extras["earth"] = deps_to_requires(earth_deps)
extras["bio"] = deps_to_requires(biology_deps)
extras["cfd"] = deps_to_requires(cfd_deps)
extras["chem"] = deps_to_requires(chemistry_deps)
extras["quantum"] = deps_to_requires(quantum_deps)
extras["dev"] = deps_to_requires(dev_deps)


extras["all"] = one_requires

setup(
    name="onescience",
    version="0.1.0-rc1",
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
    install_requires=list(basic_requires),
    python_requires=">=3.10.0",
    zip_safe=False,
)
