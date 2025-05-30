from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "cm_tpm.cpp._add",  # Output module name
        ["src/cm_tpm/cpp/add.cpp"],  # Source file
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),

    # Additional extension modules can be added here
    
]

setup(
    ext_modules=ext_modules,
)
