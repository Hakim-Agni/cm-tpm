#include <pybind11/pybind11.h>

// Define the add function
int add(int x, int y) {
    return x + y;
}

// Create Python bindings
PYBIND11_MODULE(_add, m) {
    m.def("add", &add, "A function that adds two numbers");
}
