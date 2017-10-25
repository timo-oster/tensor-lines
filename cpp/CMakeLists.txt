cmake_minimum_required(VERSION 2.8)
project(ParallelEigenvectors CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/CMake/modules)

set(PARALLELEIGENVECTORS_MAJOR_VERSION 0)
set(PARALLELEIGENVECTORS_MINOR_VERSION 0)
set(PARALLELEIGENVECTORS_PATCH_VERSION 0)
set(PARALLELEIGENVECTORS_VERSION
    ${PARALLELEIGENVECTORS_MAJOR_VERSION}.${PARALLELEIGENVECTORS_MINOR_VERSION}.${PARALLELEIGENVECTORS_PATCH_VERSION})

set(INSTALL_BIN_DIR bin CACHE PATH
    "Installation directory for executables")
set(INSTALL_LIB_DIR lib CACHE PATH
    "Installation directory for libraries")
set(INSTALL_INCLUDE_DIR include CACHE PATH
    "Installation directory for header files")

if(WIN32 AND NOT CYGWIN)
    set(DEF_INSTALL_CMAKE_DIR CMake)
else()
    set(DEF_INSTALL_CMAKE_DIR share/cmake/ParallelEigenvectors)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
    "Installation directory for CMake files")

# Make install paths absolute
foreach(p LIB BIN INCLUDE CMAKE)
    set(var INSTALL_${p}_DIR)
    if(NOT IS_ABSOLUTE "${${var}}")
        set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
    endif()
endforeach()

option(BUILD_SHARED_LIBS "Build project as shared library" OFF)

if(MSVC)
    add_definitions(-DNOMINMAX)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    add_definitions(-DNODEFAULTLIB)
    add_definitions(-DVC_EXTRALEAN)
endif()

include_directories(
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_BINARY_DIR}")

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pedantic -Wall -Wextra -Wconversion")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /W3")
endif()

add_subdirectory(src)

export(TARGETS parallel_eigenvectors
    FILE "${PROJECT_BINARY_DIR}/ParallelEigenvectorsTargets.cmake")

# export(PACKAGE ParallelEigenvectors)

file(RELATIVE_PATH REL_INCLUDE_DIR
    "${INSTALL_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}")

set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(ParallelEigenvectorsConfig.cmake.in
    "${PROJECT_BINARY_DIR}/ParallelEigenvectorsConfig.cmake" @ONLY)

set(CONF_INCLUDE_DIRS "\${PARALLELEIGENVECTORS_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(ParallelEigenvectorsConfig.cmake.in
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/ParallelEigenvectorsConfig.cmake" @ONLY)

configure_file(ParallelEigenvectorsConfigVersion.cmake.in
    "${PROJECT_BINARY_DIR}/ParallelEigenvectorsConfigVersion.cmake" @ONLY)

install(FILES
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/ParallelEigenvectorsConfig.cmake"
    "${PROJECT_BINARY_DIR}/ParallelEigenvectorsConfigVersion.cmake"
    DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

install(EXPORT ParallelEigenvectorsTargets
    DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
