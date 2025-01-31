#include <pybind11/pybind11.h>

// Define the subtract function
int subtract(int x, int y) {
    return x - y;
}

// Create Python bindings
PYBIND11_MODULE(_subtract, m) {
    m.def("subtract", &subtract, "A function that subtracts two numbers");
}
