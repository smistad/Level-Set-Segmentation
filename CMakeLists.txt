###########################################################
##          Level-Set-Segmentation project
###########################################################
project(Level-Set-Segmentation)

cmake_minimum_required(VERSION 2.8)

#------------------------------------------------------------------------------
# Set compiler flags
#------------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")


#------------------------------------------------------------------------------
# External libraries
#------------------------------------------------------------------------------
option (LS_USE_EXTRNAL_OUL "Use external OpenCLUtilityLibrary" OFF)
option (LS_EXTERNAL_OUL_PATH "Path to external OpenCLUtilityLibrary build" "")
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/CMake
    )
include (lsInitializeLibraries)
ls_initialize_external_libraries()


#------------------------------------------------------------------------------
# Where to look for includes and libraries
#------------------------------------------------------------------------------
include_directories(${PROJECT_BINARY_DIR})


#------------------------------------------------------------------------------
# Targets
#------------------------------------------------------------------------------
add_executable(levelSetSeg main.cpp levelSet.cpp)
target_link_libraries(levelSetSeg SIPL OpenCLUtilityLibrary OpenCL)

add_library(levelSetSegLib levelSet.cpp)
target_link_libraries(levelSetSegLib SIPL OpenCLUtilityLibrary)

#------------------------------------------------------------------------------
# Configure file for find_package module 
#------------------------------------------------------------------------------
set( KERNELS_DIR ${PROJECT_SOURCE_DIR})

set( Level-Set-Segmentation_INCLUDE_DIRS
    ${Level-Set-Segmentation_SOURCE_DIR}
) 

set( Level-Set-Segmentation_LIBRARY_DIRS
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
