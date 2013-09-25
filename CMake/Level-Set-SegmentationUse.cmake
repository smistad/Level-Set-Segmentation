###########################################################
##          Level-Set-Segmentation Use file
###########################################################

#------------------------------------------------------------------------------
# External libraries
#------------------------------------------------------------------------------

# GTK
#find_package (PkgConfig REQUIRED)
#pkg_check_modules (GTK2 REQUIRED gtk+-2.0 gthread-2.0)

# SIPL
find_package(SIPL PATHS "${Level-Set-Segmentation_BINARY_DIR}/SIPL" REQUIRED)
include(${SIPL_USE_FILE})

# OpenCLUtilities
find_package(OCL-Utilities PATHS "${Level-Set-Segmentation_BINARY_DIR}/OpenCLUtilities" REQUIRED)
include(${OCL-Utilities_USE_FILE})

#------------------------------------------------------------------------------
# Where to look for includes and libraries
#------------------------------------------------------------------------------
include_directories( ${Level-Set-Segmentation_INCLUDE_DIRS}  ${Level-Set-Segmentation_BINARY_DIR})
link_directories (${Level-Set-Segmentation_LIBRARY_DIRS})

