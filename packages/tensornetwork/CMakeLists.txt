cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# Find Python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Find LAPACK
find_package(LAPACK REQUIRED)

# Find BLAS 
find_package(BLAS REQUIRED)

# Try to find LAPACKE header
find_path(LAPACKE_INCLUDE_DIR
  NAMES lapacke.h
  PATHS 
    /opt/homebrew/include
    /opt/homebrew/Cellar/openblas/*/include
    /usr/include
    /usr/local/include
    ${LAPACK_INCLUDE_DIRS}
)

if(NOT LAPACKE_INCLUDE_DIR)
  message(FATAL_ERROR "LAPACKE header not found. Please install LAPACK development package (try: brew install openblas).")
endif()

# Try to find CBLAS header  
find_path(CBLAS_INCLUDE_DIR
  NAMES cblas.h
  PATHS
    /opt/homebrew/include
    /opt/homebrew/Cellar/openblas/*/include
    /usr/include
    /usr/local/include
    ${BLAS_INCLUDE_DIRS}
)

if(NOT CBLAS_INCLUDE_DIR)
  message(FATAL_ERROR "CBLAS header not found. Please install BLAS development package (try: brew install openblas).")
endif()

# Create the pybind11 module
pybind11_add_module(libincrementalqr MODULE src/incrementalqr.cpp)

# Set target properties
target_include_directories(libincrementalqr PRIVATE 
  ${LAPACKE_INCLUDE_DIR} 
  ${CBLAS_INCLUDE_DIR}
)

target_link_libraries(libincrementalqr PRIVATE 
  ${LAPACK_LIBRARIES} 
  ${BLAS_LIBRARIES}
)

# Compiler-specific options
target_compile_definitions(libincrementalqr PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Install the module
install(TARGETS libincrementalqr DESTINATION ${SKBUILD_PROJECT_NAME})