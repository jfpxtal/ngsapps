project(ngsapps)
cmake_minimum_required(VERSION 3.1)

find_package(NGSolve CONFIG REQUIRED
  HINTS $ENV{NETGENDIR}/.. /opt/netgen/ /Applications/Netgen.app/Contents/Resources/CMake C:/netgen
)

# check if CMAKE_INSTALL_PREFIX is set by user, if not install in NGSolve python dir
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${NGSOLVE_INSTALL_DIR}/${NGSOLVE_INSTALL_DIR_PYTHON}/ngsapps CACHE PATH "Install dir" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

SET(CMAKE_CXX_COMPILER "${NGSOLVE_INSTALL_DIR}/${NGSOLVE_INSTALL_DIR_BIN}/ngscxx")

add_subdirectory(utils)
