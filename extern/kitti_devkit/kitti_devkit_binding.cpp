#include <pybind11/pybind11.h>

#include "cpp/evaluate_odometry.cpp"
#include "cpp/matrix.cpp"

namespace py = pybind11;

PYBIND11_MODULE(kitti_devkit_, m)
{
  m.def("eval", &eval, py::arg("gt_dir"), py::arg("pred_dir"));
}
