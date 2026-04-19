add_executable(bench_matrix_multiply matrix_multiply.cpp)

target_include_directories(bench_matrix_multiply
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)

find_package(OpenMP REQUIRED)

target_compile_options(bench_matrix_multiply PRIVATE
    $<IF:$<CXX_COMPILER_ID:MSVC>,/O2 /arch:AVX2 /openmp,-O3 -mavx2 -mfma>
)

target_link_libraries(bench_matrix_multiply PRIVATE OpenMP::OpenMP_CXX)

