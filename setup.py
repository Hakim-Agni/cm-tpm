from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "cm_tpm.cpp._add",  # Output module name
        ["src/cm_tpm/cpp/add.cpp"],  # Source file
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),
    Extension(
        "cm_tpm.cpp._multiply",  # Output module name
        ["src/cm_tpm/cpp/multiply.cpp"],  # Source file
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),
    Extension(
        "cm_tpm.cpp._subtract",  # Output module name
        ["src/cm_tpm/cpp/subtract.cpp"],  # Source file
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),
    Extension(
        "cm_tpm.cpp._divide",  # Output module name
        ["src/cm_tpm/cpp/divide.cpp"],  # Source file
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),
]

setup(
    ext_modules=ext_modules,
)