add_executable(tensor_test
    tensor_test.cpp
)

target_include_directories(tensor_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if(MSVC)
    target_compile_options(tensor_test PRIVATE /W4 /permissive-)
else()
    target_compile_options(tensor_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME tensor_test COMMAND tensor_test)

add_executable(linear_ops_test
    linear_ops_test.cpp
)

target_include_directories(linear_ops_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if(MSVC)
    target_compile_options(linear_ops_test PRIVATE /W4 /permissive-)
else()
    target_compile_options(linear_ops_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME linear_ops_test COMMAND linear_ops_test)

add_executable(activation_test
    activation_test.cpp
)

target_include_directories(activation_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if(MSVC)
    target_compile_options(activation_test PRIVATE /W4 /permissive-)
else()
    target_compile_options(activation_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME activation_test COMMAND activation_test)

add_executable(loss_test
    loss_test.cpp
)

target_include_directories(loss_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if(MSVC)
    target_compile_options(loss_test PRIVATE /W4 /permissive-)
else()
    target_compile_options(loss_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME loss_test COMMAND loss_test)

add_executable(layers_test
    layers_test.cpp
)

target_include_directories(layers_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if(MSVC)
    target_compile_options(layers_test PRIVATE /W4 /permissive-)
else()
    target_compile_options(layers_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME layers_test COMMAND layers_test)

add_executable(optim_test
    optim_test.cpp
)

target_include_directories(optim_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if(MSVC)
    target_compile_options(optim_test PRIVATE /W4 /permissive-)
else()
    target_compile_options(optim_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME optim_test COMMAND optim_test)
