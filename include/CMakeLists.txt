find_package(Armadillo REQUIRED)
add_library(Network network.cpp)

# state that anybody linking to us needs to include the current source dir
# to find MathFunctions.h, while we don't.
target_link_libraries(Network PUBLIC ${ARMADILLO_LIBRARIES})
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_include_directories(
  Network
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)
