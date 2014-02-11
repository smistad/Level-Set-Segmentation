###########################################################
##          Level-Set-Segmentation Use file
###########################################################

#------------------------------------------------------------------------------
# Options
#------------------------------------------------------------------------------
option (LS_USE_EXTRNAL_OUL "Use external OpenCLUtilityLibrary" OFF)

#------------------------------------------------------------------------------
# External libraries
#------------------------------------------------------------------------------

# SIPL
find_package(SIPL PATHS "${Level-Set-Segmentation_BINARY_DIR}/SIPL" REQUIRED)
include(${SIPL_USE_FILE})

# OpenCLUtilityLibrary
if(LS_USE_EXTRNAL_OUL)
    message(STATUS "Using external use file for OpenCLUtilityLibrary in LS: "${TSF_EXTERNAL_OUL_USEFILE})
    include(${LS_EXTERNAL_OUL_USEFILE})
else(LS_USE_EXTRNAL_OUL)
    message(STATUS "Using submodule for OpenCLUtility in LS")
    find_package(OpenCLUtilityLibrary PATHS "${Level-Set-Segmentation_BINARY_DIR}/OpenCLUtilityLibrary" REQUIRED)
    include(${OpenCLUtilityLibrary_USE_FILE})
endif(LS_USE_EXTRNAL_OUL)

#------------------------------------------------------------------------------
# Where to look for includes and libraries
#------------------------------------------------------------------------------
include_directories( ${Level-Set-Segmentation_INCLUDE_DIRS}  ${Level-Set-Segmentation_BINARY_DIR})
link_directories (${Level-Set-Segmentation_LIBRARY_DIRS})
