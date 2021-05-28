from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension


SOURCES = [
    'kitti_devkit_binding.cpp',
]

ext_modules = [
    Pybind11Extension('kitti_devkit_', sources=SOURCES)
]

setup(
    name='kitti_devkit',
    version='0.0.1',
    description='KITTI Devkit Python Binding',
    author='Markus Horn',
    author_email='markus.hn11@gmail.com',
    packages=['kitti_devkit'],
    ext_modules=ext_modules,
    install_requires=[
        'pybind11==2.6.2',
    ]
)
