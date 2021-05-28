from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension


SOURCES = [
    'gicp_binding.cpp',
    'gicp/optimize.cpp',
    'gicp/gicp.cpp',
    'gicp/bfgs_funcs.cpp',
    'gicp/scan.cpp',
    'gicp/transform.cpp',
    'gicp/ann_1.1.1/src/ANN.cpp',
    'gicp/ann_1.1.1/src/bd_fix_rad_search.cpp',
    'gicp/ann_1.1.1/src/bd_pr_search.cpp',
    'gicp/ann_1.1.1/src/bd_search.cpp',
    'gicp/ann_1.1.1/src/bd_tree.cpp',
    'gicp/ann_1.1.1/src/brute.cpp',
    'gicp/ann_1.1.1/src/kd_dump.cpp',
    'gicp/ann_1.1.1/src/kd_fix_rad_search.cpp',
    'gicp/ann_1.1.1/src/kd_pr_search.cpp',
    'gicp/ann_1.1.1/src/kd_search.cpp',
    'gicp/ann_1.1.1/src/kd_split.cpp',
    'gicp/ann_1.1.1/src/kd_tree.cpp',
    'gicp/ann_1.1.1/src/kd_util.cpp',
    'gicp/ann_1.1.1/src/perf.cpp'
]

INCLUDE_DIRS = [
    'gicp',
    'gicp/ann_1.1.1/include',
    'gicp/ann_1.1.1/include/ANN'
]

LIBRARIES = [
    'gsl',
    'gslcblas',
    'stdc++'
]

ext_modules = [
    Pybind11Extension('gicp', sources=SOURCES, include_dirs=INCLUDE_DIRS, libraries=LIBRARIES)
]

setup(
    name='gicp',
    version='0.0.1',
    description='GICP Python Binding',
    author='Markus Horn',
    author_email='markus.hn11@gmail.com',
    ext_modules=ext_modules,
    install_requires=[
        'pybind11==2.6.2',
    ]
)
