add_executable(bench_matrix_multiply matrix_multiply.cpp)

target_include_directories(bench_matrix_multiply
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(bench_matrix_multiply PRIVATE
    $<IF:$<CXX_COMPILER_ID:MSVC>,/O2 /arch:AVX2,-O3 -mavx2 -mfma>
)

