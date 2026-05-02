cmake_minimum_required(VERSION 3.16)

project(dfml VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


add_executable(dfml
    src/main.cpp
)

# Add the parent include directory so #include "dfml/..." works everywhere
target_include_directories(dfml
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(OpenMP REQUIRED)
target_link_libraries(dfml PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
	target_compile_options(dfml PRIVATE /W4 /permissive- /O2 /arch:AVX2)
else()
	target_compile_options(dfml PRIVATE -Wall -Wextra -Wpedantic -O3 -mavx2 -mfma)
endif()

include(CTest)
if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
	add_subdirectory(tests)
endif()

option(BUILD_BENCHMARKS "Build benchmarks" OFF)
if(BUILD_BENCHMARKS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CMakeLists.txt")
    add_subdirectory(benchmarks)
endif()