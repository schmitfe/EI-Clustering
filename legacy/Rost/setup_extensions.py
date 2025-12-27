from pathlib import Path

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np


ROOT = Path(__file__).parent

ext_modules = [
    Extension(
        name="Cspiketools",
        sources=[str(ROOT / "code" / "Cspiketools.pyx")],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="BiNet.cstuff",
        sources=[str(ROOT / "code" / "BiNet" / "src" / "BiNet" / "cstuff.pyx")],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="rost_extensions",
    package_dir={"BiNet": "code/BiNet/src/BiNet", "": "code"},
    ext_modules=cythonize(ext_modules, language_level=3),
)
