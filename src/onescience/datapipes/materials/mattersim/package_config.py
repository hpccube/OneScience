from pathlib import Path

import numpy as np
from setuptools import Extension


def get_package_data():
    return {
        "onescience.datapipes.materials.mattersim": ["threebody_indices.pyx"],
        "onescience.models.mattersim": ["LICENSE.txt", "SOURCE.md"],
    }


def get_extensions(project_root):
    from Cython.Build import cythonize

    source = (
        Path(project_root)
        / "src/onescience/datapipes/materials/mattersim/threebody_indices.pyx"
    )
    extension = Extension(
        "onescience.datapipes.materials.mattersim.threebody_indices",
        [str(source)],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_API_VERSION")],
    )
    return cythonize(
        [extension],
        build_dir=str(Path(project_root) / "build/cython"),
        compiler_directives={"language_level": "3"},
    )
