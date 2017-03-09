# - Try to find CTPL
# Once done, this will define
#
#  CTPL_FOUND - system has CTPL
#  CTPL_INCLUDE_DIRS - the CTPL include directories
#  CTPL_COMPILER_FLAGS - the compiler flags needed to use CTPL
#  CTPL_LINKER_FLAGS - the linker flags needed to use CTPL

include(LibFindMacros)

# Include dir
find_path(CTPL_INCLUDE_DIR
  NAMES ctpl.h ctpl_stl.h
)

set(CTPL_PROCESS_INCLUDES CTPL_INCLUDE_DIR)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CTPL_COMPILER_FLAGS "-pthread")
    set(CTPL_LINKER_FLAGS "-pthread")
endif()

find_package(Boost COMPONENTS lockfree)
if(Boost_FOUND)
    set(CTPL_PROCESS_INCLUDES ${CTPL_PROCESS_INCLUDES} ${Boost_INCLUDE_DIRS})
endif()

libfind_process(CTPL)