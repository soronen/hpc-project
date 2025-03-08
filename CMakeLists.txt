cmake_minimum_required(VERSION 3.10)
project(hpc-baseline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Define option to enable LUMI/ROCm specific settings
option(USE_LUMI "Build with LUMI/ROCm support" ON)


# LUMI/ROCm specific settings
if(USE_LUMI)
    message(STATUS "Configuring for LUMI with ROCm")
    include_directories(/opt/rocm/include)
    link_directories(/opt/rocm/lib)
    # ROCm OpenCL implementation workarounds
    add_definitions(-Dinline=)
    add_definitions(-DUSE_LUMI)
endif()

# Windows OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP. Adding flags: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Mac OpenMP
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/opt/libomp/lib -lomp")
endif()

# Common flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native")
endif()

option(USE_MPI "Build with MPI support" OFF)

if(USE_MPI)
    find_package(MPI REQUIRED)
    include_directories(${MPI_INCLUDE_PATH})
    add_definitions(-DUSE_MPI)
    set(EXTRA_LIBS ${EXTRA_LIBS} ${MPI_LIBRARIES})
endif()

# OpenCL configuration with custom paths for LUMI/ROCm
find_package(OpenCL REQUIRED)

set(SOURCES
    main.cc
    bvh.cc
    mesh.cc
    bmp.cc
    scene.cc
    opencl_utils.cc
)

add_executable(pt ${SOURCES})
target_include_directories(pt PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(pt PRIVATE ${OpenCL_LIBRARIES} ${EXTRA_LIBS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(pt PRIVATE OpenMP::OpenMP_CXX)
endif()