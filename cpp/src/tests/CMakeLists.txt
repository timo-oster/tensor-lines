option(BUILD_TESTS "Build the unit tests" OFF)

option(RUN_TESTS "Run the unit tests after each build" OFF)

if(${BUILD_TESTS})
    add_executable(unit_tests UnitTests.cpp)
    set_property(TARGET unit_tests PROPERTY CXX_STANDARD 14)
    if(${RUN_TESTS})
        add_custom_target(tests ALL
                          COMMAND unit_tests
                          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                          COMMENT "Run Unit Tests")
        add_dependencies(tests unit_tests)
    endif()
endif()