set(PEV_SOURCES
        main.cc
        ParallelEigenvectors.cc
        ParallelEigenvectorsEvaluator.cc
        TensorSujudiHaimesEvaluator.cc
        vtkParallelEigenvectors.cc)

set(PEV_HEADERS
    utils.hh
    ParallelEigenvectors.hh
    ParallelEigenvectorsEvaluator.hh
    TensorSujudiHaimesEvaluator.hh
    vtkParallelEigenvectors.hh
    TensorProductBezierTriangle.hh)

source_group("Header Files" FILES ${PEV_HEADERS})
source_group("Source Files" FILES ${PEV_SOURCES})

find_package(PythonInterp 3 REQUIRED)

set(TPBT_COLLECTION_HEADER TensorProductBezierTriangles.hh)
set(HEADER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/generate_collection_header.py)
set(HEADER_GEN ${PYTHON_EXECUTABLE} ${HEADER_SCRIPT} -o ${TPBT_COLLECTION_HEADER})

set(TPBT_GEN_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/generate_tpbt.py)
set(GENERATOR1 ${PYTHON_EXECUTABLE} ${TPBT_GEN_SCRIPT} 3)
set(GENERATOR2 ${PYTHON_EXECUTABLE} ${TPBT_GEN_SCRIPT} 2 3)

execute_process(COMMAND ${GENERATOR1} -p
  OUTPUT_VARIABLE GENERATED_SOURCES1)
execute_process(COMMAND ${GENERATOR2} -p
  OUTPUT_VARIABLE GENERATED_SOURCES2)
string(REPLACE "\n" ";" GENERATED_SOURCES1 ${GENERATED_SOURCES1})
string(REPLACE "\n" ";" GENERATED_SOURCES2 ${GENERATED_SOURCES2})
set(GENERATED_SOURCES ${GENERATED_SOURCES1} ${GENERATED_SOURCES2})

execute_process(COMMAND ${CMAKE} -E echo ${GENERATED_SOURCES})

add_custom_command(OUTPUT ${TPBT_COLLECTION_HEADER}
                   COMMAND ${HEADER_GEN} ${GENERATED_SOURCES}
                   DEPENDS ${HEADER_SCRIPT}
                   COMMENT Generate collection header for specializations of TensorProductBezierTriangle)

add_custom_command(OUTPUT ${GENERATED_SOURCES1}
  COMMAND ${GENERATOR1}
  DEPENDS ${TPBT_GEN_SCRIPT}
  COMMENT "Generate specializations of TensorProductBezierTriangle for 1 space")
add_custom_command(OUTPUT ${GENERATED_SOURCES2}
  COMMAND ${GENERATOR2}
  DEPENDS ${TPBT_GEN_SCRIPT}
  COMMENT "Generate specializations of TensorProductBezierTriangle for 2 spaces")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(parallel_eigenvectors ${PEV_SOURCES} ${GENERATED_SOURCES} ${TPBT_COLLECTION_HEADER})
add_executable(generate_tet_dataset generate_tet_dataset.cc)
add_executable(generate_grid_dataset generate_grid_dataset.cc)
add_executable(filter_tensor_lines filter_lines.cc)

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
    target_link_libraries(filter_tensor_lines ${Boost_LIBRARIES})
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
    target_link_libraries(filter_tensor_lines ${VTK_LIBRARIES})
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

install(TARGETS filter_tensor_lines
    EXPORT ParallelEigenvectorsTargets
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)

add_subdirectory(tests)
