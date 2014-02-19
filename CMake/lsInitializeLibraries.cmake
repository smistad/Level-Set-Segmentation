###############################################################################
# Initialize external libraries for LS
#
#
###############################################################################
macro(ls_initialize_external_libraries)
    message( STATUS "Initializing External Libraries" )
    ls_initialize_oul()
    add_subdirectory(SIPL)
    ls_initialize_sipl()
endmacro(ls_initialize_external_libraries)


###############################################################################
# Initialize SIPL library
#
###############################################################################
macro(ls_initialize_sipl)
    find_package(SIPL PATHS "${Level-Set-Segmentation_BINARY_DIR}/SIPL" REQUIRED)
    include(${SIPL_USE_FILE})
endmacro(ls_initialize_sipl)

###############################################################################
# Initialize OpenCLUtility library
#
# Uses predefined variables:
#    TSF_USE_EXTRNAL_OUL : path to oul
#    TSF_EXTERNAL_OUL_PATH : path to oul build dir
#
###############################################################################
macro(ls_initialize_oul)
    if(LS_USE_EXTRNAL_OUL)
        message(STATUS "Using external OpenCLUtilityLibrary in LS.")
        find_package(OpenCLUtilityLibrary PATHS ${LS_EXTERNAL_OUL_PATH})
        message(STATUS "OpenCLUtilityLibrary_USE_FILE "${OpenCLUtilityLibrary_USE_FILE})
        include(${OpenCLUtilityLibrary_USE_FILE})
    else(LS_USE_EXTRNAL_OUL)
        message(STATUS "Using submodule for OpenCLUtility in LS")
        add_subdirectory(OpenCLUtilityLibrary)
        find_package(OpenCLUtilityLibrary PATHS "${Level-Set-Segmentation_BINARY_DIR}/OpenCLUtilityLibrary" REQUIRED)
        include(${OpenCLUtilityLibrary_USE_FILE})
    endif(LS_USE_EXTRNAL_OUL)
endmacro(ls_initialize_oul)
