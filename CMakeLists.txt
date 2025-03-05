cmake_minimum_required(VERSION 3.10)
project(hpc-baseline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Let CMake find the right flags to link against OpenMP (both on Mac and Windows).
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP. Adding flags: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# If you want additional optimization flags, set them conditionally:
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Common flags for Clang on both Mac and Windows
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native")
endif()

find_package(OpenCL REQUIRED)

set(SOURCES
    main.cc
    bvh.cc
    mesh.cc
    bmp.cc
    scene.cc
    opencl_utils.cpp
)

add_executable(pt ${SOURCES})
target_include_directories(pt PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(pt PRIVATE ${OpenCL_LIBRARIES})

if(OpenMP_CXX_FOUND)
    # Use the imported target. This is the modern way to link OpenMP in CMake
    target_link_libraries(pt PRIVATE OpenMP::OpenMP_CXX)
endif()

# On Windows, no separate 'm' library is needed.
# On Unix-like systems, we often link -lm:
if(NOT WIN32)
    target_link_libraries(pt PRIVATE m)
endif()
