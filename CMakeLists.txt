cmake_minimum_required(VERSION 2.8)
project(Level-Set-Segmentation)



add_subdirectory(SIPL)
find_package(SIPL PATHS "${Level-Set-Segmentation_BINARY_DIR}/SIPL" REQUIRED)
include(${SIPL_USE_FILE})

add_subdirectory(OpenCLUtilityLibrary)
find_package(OpenCLUtilityLibrary PATHS "${Level-Set-Segmentation_BINARY_DIR}/OpenCLUtilityLibrary" REQUIRED)
include(${OpenCLUtilityLibrary_USE_FILE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")

include_directories(${PROJECT_BINARY_DIR})

add_executable(levelSetSeg main.cpp levelSet.cpp)
target_link_libraries(levelSetSeg SIPL OpenCLUtilityLibrary)

add_library(levelSetSegLib levelSet.cpp)
target_link_libraries(levelSetSegLib SIPL OpenCLUtilityLibrary)

set(KERNELS_DIR ${PROJECT_SOURCE_DIR})

set( Level-Set-Segmentation_INCLUDE_DIRS
    ${Level-Set-Segmentation_SOURCE_DIR}
) 

set(Level-Set-Segmentation_LIBRARY_DIRS
    ${Level-Set-Segmentation_BINARY_DIR}
    ${OCL-Utilities_LIBRARY_DIRS}
    ${SIPL_BINARY_LIBRARY_DIRS}
)    

configure_file (
    "${PROJECT_SOURCE_DIR}/CMake/Level-Set-SegmentationConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/Level-Set-SegmentationConfig.cmake"
)
 
#------------------------------------------------------------------------------
# Configure file for settings, filepaths, parameters ...
#------------------------------------------------------------------------------

configure_file(
 "${PROJECT_SOURCE_DIR}/CMake/level-set-segmentation-config.h.in"
 "${PROJECT_BINARY_DIR}/config.h"
)
