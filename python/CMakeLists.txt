cmake_minimum_required(VERSION 3.5)
project(pymcc_lidar)

find_package(PythonInterp)
find_package(PythonLibs)

find_path(BOOST_DIR "boost")

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )

include(UseCython)
include(FindNumPy)

link_directories(../libmcc_lidar/lib/)
include_directories(../libmcc_lidar/include/)
include_directories("${BOOST_DIR}")
# Use BEFORE so the venv numpy headers take precedence over any system numpy
# headers that may exist under /usr/include/python3.x/numpy/
include_directories(BEFORE "${PYTHON_NUMPY_INCLUDE_DIR}")
include_directories("./")

set_source_files_properties(pymcc_lidar.pyx PROPERTIES CYTHON_IS_CXX 1)
# Explicitly set module name to avoid Cython deriving an invalid name from
# pip's temporary build directories (which may contain dashes)
set(CYTHON_FLAGS "--module-name=pymcc_lidar")
cython_add_module(pymcc_lidar pymcc_lidar.pyx pymccp.cpp)
target_link_libraries(pymcc_lidar mcc_lidar)
install(TARGETS pymcc_lidar DESTINATION .)

