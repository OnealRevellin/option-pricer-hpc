cmake_minimum_required(VERSION 3.16)
project(option_pricer_hpc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -funroll-loops -fopenmp")

# Enable testing
enable_testing()

# Find OpenMP
find_package(OpenMP REQUIRED)

# ------------------------------------------------------------------------------
# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ------------------------------------------------------------------------------
# Add core library target (source files from cpp/)
add_library(option_pricer_lib
    cpp/vanilla_options_pricer.cpp
    cpp/models/gbsm.cpp
)

target_include_directories(option_pricer_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/models
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(option_pricer_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# ------------------------------------------------------------------------------
# Main executable
add_executable(pricer cpp/main.cpp)
target_link_libraries(pricer PRIVATE option_pricer_lib)

# ------------------------------------------------------------------------------
# Tests
add_executable(test_gbsm tests/cpp/test_gbsm.cpp)

target_include_directories(test_gbsm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/models
)

target_link_libraries(test_gbsm
    PRIVATE
    option_pricer_lib
    gtest
    gtest_main
    pthread
)

add_test(NAME GBSMTest COMMAND test_gbsm)
