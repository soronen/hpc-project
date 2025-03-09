cmake_minimum_required(VERSION 3.10)
project(hpc-baseline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

option(USE_LUMI "Build with LUMI/ROCm support" OFF)

if(USE_LUMI)
    message(STATUS "Configuring for LUMI with ROCm")
    include_directories(/opt/rocm/include)
    link_directories(/opt/rocm/lib)
    add_definitions(-DUSE_LUMI)

    find_package(MPI REQUIRED)
    include_directories(${MPI_INCLUDE_PATH})
    add_definitions(-DUSE_MPI)
    set(EXTRA_LIBS ${EXTRA_LIBS} ${MPI_LIBRARIES})
    
    # Manually set OpenCL library for LUMI/ROCm
    set(OpenCL_LIBRARIES OpenCL)
else()
    message(STATUS "Configuring for generic OpenCL")
    find_package(OpenCL REQUIRED)
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

# Common flags for Clang and GNU (GCC)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(STATUS "Found Clang or GNU compiler. Adding flags: -O3 -ffast-math -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native")
endif()



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
