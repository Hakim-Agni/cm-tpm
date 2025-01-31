from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "cm_tpm._add",  # Output module name
        ["src/cm_tpm/add.cpp"],  # Source file
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),
    Extension(
        "cm_tpm._multiply",  # Output module name
        ["src/cm_tpm/multiply.cpp"],  # Source file
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),
]

setup(
    ext_modules=ext_modules,
)