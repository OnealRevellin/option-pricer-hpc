cmake_minimum_required(VERSION 3.16)
project(option_pricer_hpc LANGUAGES CXX)

# Enable C++20 standard.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable compiler optimization flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -funroll-loops -fopenmp")

# Add the main executable.
add_executable(pricer
    main.cpp
    vanilla_options_pricer.cpp
    models/gbsm.cpp
)

# Include directories.
target_include_directories(pricer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/models
)

# Link OpenMP if found.
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pricer PUBLIC OpenMP::OpenMP_CXX)
endif()