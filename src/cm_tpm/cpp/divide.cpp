#include <pybind11/pybind11.h>

int divide(int a, int b) {
    if (b == 0) throw std::runtime_error("Division by zero!");
    return a / b;
}

// Create Python bindings
PYBIND11_MODULE(_divide, m) {
    m.def("divide", &divide, "A function that divides two numbers");
}