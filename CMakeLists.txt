cmake_minimum_required(VERSION 3.22.3)

# Set the project name
project(Pantagruel-CPP VERSION 0.0.1)

# Export compile commands for the LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the c++ version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# Specify directories
find_package(Armadillo REQUIRED)
# add_library(PANTAGRUEL ${PROJECT_SOURCE_DIR}/src/pantagruel.cpp)

# add the executable
add_executable(net main.cpp)
add_subdirectory(include)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Link libraries
target_link_libraries(net PUBLIC Network)
