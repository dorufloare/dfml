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

if(MSVC)
	target_compile_options(dfml PRIVATE /W4 /permissive-)
else()
	target_compile_options(dfml PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
	add_subdirectory(tests)
endif()
