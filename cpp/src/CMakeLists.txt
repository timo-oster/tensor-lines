set(PEV_SOURCES
        main.cc
        ParallelEigenvectors.cc
        vtkParallelEigenvectors.cc)

set(PEV_HEADERS
    ParallelEigenvectors.hh
    utils.hh
    vtkParallelEigenvectors.hh
    TensorProductBezierTriangle.hh)

source_group("Header Files" FILES ${PEV_HEADERS})
source_group("Source Files" FILES ${PEV_SOURCES})

add_executable(parallel_eigenvectors ${PEV_SOURCES} ${PEV_HEADERS})
add_executable(generate_tet_dataset generate_tet_dataset.cc)
add_executable(generate_grid_dataset generate_grid_dataset.cc)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()


find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
endif()

if(WIN32)
    set(Boost_USE_STATIC_LIBS OFF)
    set(Boost_USE_MULTITHREADED ON)
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(Boost REQUIRED COMPONENTS program_options)
if(Boost_FOUND)
    target_link_libraries(parallel_eigenvectors ${Boost_LIBRARIES})
    target_link_libraries(generate_tet_dataset ${Boost_LIBRARIES})
    target_link_libraries(generate_grid_dataset ${Boost_LIBRARIES})
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
    if(WIN32)
        add_definitions(-DBOOST_ALL_DYN_LINK)
    endif()
endif()

find_package(VTK COMPONENTS vtkFiltersModeling vtkIOLegacy)
if(VTK_FOUND)
    include(${VTK_USE_FILE})
    target_link_libraries(parallel_eigenvectors ${VTK_LIBRARIES})
    target_link_libraries(generate_tet_dataset ${VTK_LIBRARIES})
    target_link_libraries(generate_grid_dataset ${VTK_LIBRARIES})
endif()

option(DRAW_DEBUG "Visualize the recursion algorithm (needs CImg)" OFF)
if(DRAW_DEBUG)
    find_package(CImg REQUIRED)
    if(CIMG_FOUND)
        include_directories(SYSTEM ${CIMG_INCLUDE_DIRS})
        target_link_libraries(parallel_eigenvectors ${CIMG_LIBRARIES})
        target_compile_definitions(parallel_eigenvectors PRIVATE "-DDRAW_DEBUG")
    endif()
endif()

install(TARGETS parallel_eigenvectors
    EXPORT ParallelEigenvectorsTargets
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)

install(TARGETS generate_tet_dataset
    EXPORT ParallelEigenvectorsTargets
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)

install(TARGETS generate_grid_dataset
    EXPORT ParallelEigenvectorsTargets
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)
