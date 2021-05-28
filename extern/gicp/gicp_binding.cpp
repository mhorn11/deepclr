#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <gicp.h>

namespace py = pybind11;

dgc::gicp::GICPPointSet prepare(std::vector<std::array<double, 3>> cloud)
{
  // empty point
  dgc::gicp::GICPPoint pt;
  pt.range = -1;
  for(int k = 0; k < 3; k++) {
    for(int l = 0; l < 3; l++) {
      pt.C[k][l] = (k == l) ? 1:0;
    }
  }

  // copy points
  dgc::gicp::GICPPointSet point_set;
  for(size_t i = 0; i < cloud.size(); ++i) {
    pt.x = cloud.at(i)[0];
    pt.y = cloud.at(i)[1];
    pt.z = cloud.at(i)[2];
    point_set.AppendPoint(pt);
  }

  return point_set;
}

std::array<std::array<double, 4>, 4> gicp(
    dgc::gicp::GICPPointSet cloud1,
    dgc::gicp::GICPPointSet cloud2,
    double epsilon=1e-3,
    double max_distance=5.0)
{
  // set up the transformations
  dgc_transform_t t_base, t0, t1;
  dgc_transform_identity(t_base);
  dgc_transform_identity(t0);
  dgc_transform_identity(t1);

  // settings
  cloud1.SetGICPEpsilon(epsilon);
  cloud1.BuildKDTree();
  cloud1.ComputeMatrices();

  cloud2.SetGICPEpsilon(epsilon);
  cloud2.BuildKDTree();
  cloud2.ComputeMatrices();

  // align the point clouds
  dgc_transform_copy(t1, t0);
  cloud2.SetMaxIterationInner(8);
  cloud2.SetMaxIteration(100);

  int iterations = cloud2.AlignScan(&cloud1, t_base, t1, max_distance);

  // copy result to vector
  std::array<std::array<double, 4>, 4> m;
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      m[i][j] = t1[i][j];
    }
  }

  return m;
}

PYBIND11_MODULE(gicp, m)
{
  py::class_<dgc::gicp::GICPPointSet>(m, "GICPPointSet");
  m.def("prepare", &prepare,
        py::arg("cloud"));
  m.def("gicp", &gicp,
        py::arg("cloud1"), py::arg("cloud2"), py::arg("epsilon") = 1e-3, py::arg("max_distance") = 5.0);
}
