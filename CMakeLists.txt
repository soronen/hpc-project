cmake_minimum_required(VERSION 3.10)
project(hpc-baseline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native")

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Enable OpenMP support with Clang on macOS
    include_directories(/opt/homebrew/opt/libomp/include)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")
endif()


set(SOURCES
    main.cc
    bvh.cc
    mesh.cc
    bmp.cc
    scene.cc
)

add_executable(pt ${SOURCES})
target_link_libraries(pt m)