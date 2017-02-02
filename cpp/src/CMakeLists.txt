add_executable(parallel_eigenvectors main.cc)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

if(WIN32)
    set(Boost_USE_STATIC_LIBS OFF)
    set(Boost_USE_MULTITHREADED ON)
endif()

find_package(Boost REQUIRED)
if(Boost_FOUND)
    target_link_libraries(parallel_eigenvectors ${Boost_LIBRARIES})
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
    if(WIN32)
        add_definitions(-DBOOST_ALL_DYN_LINK)
    endif()
endif()

find_package(CImg REQUIRED)
if(CIMG_FOUND)
    include_directories(SYSTEM ${CIMG_INCLUDE_DIRS})
    target_link_libraries(parallel_eigenvectors ${CIMG_LIBRARIES})
endif()

find_package(VTK REQUIRED)
if(VTK_FOUND)
    include(${VTK_USE_FILE})
    target_link_libraries(parallel_eigenvectors ${VTK_LIBRARIES})
endif()

install(TARGETS parallel_eigenvectors
    EXPORT ParallelEigenvectorsTargets
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)
