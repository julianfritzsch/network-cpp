cmake_minimum_required(VERSION 3.22.3)

# Set the project name
project(Network-CPP VERSION 0.0.1)

# Export config header
configure_file(ProjectConfig.hpp.in ProjectConfig.hpp)

# Export compile commands for the LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the c++ version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")
add_compile_options(-Wall)

# Specify directories
find_package(Armadillo REQUIRED)
find_package(Matplot++ REQUIRED)


# add the executable
add_executable(net main.cpp)
add_subdirectory(include)
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_include_directories(net PUBLIC "${PROJECT_BINARY_DIR}")

set_source_files_properties(main.cpp PROPERTIES COMPILE_OPTIONS "-Wall")
# Link libraries
target_link_libraries(net PUBLIC Network Matplot++::matplot)

