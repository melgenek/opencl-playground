cmake_minimum_required(VERSION 3.7)
project(opencl_playground VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-march=native -ffast-math -funroll-loops)
endif ()

add_executable(hands_on_ex1_c hands_on/ex1/main.c common/err_code.h)
target_link_libraries(hands_on_ex1_c OpenCL::OpenCL)

add_executable(hands_on_ex1 hands_on/ex1/main.cpp common/cpp/cl.hpp common/err_code.h)
target_link_libraries(hands_on_ex1 OpenCL::OpenCL)

add_executable(hands_on_ex2_3_c hands_on/ex2_3/main.c common/err_code.h common/c/wtime.c common/c/device_info.c)
target_link_libraries(hands_on_ex2_3_c OpenCL::OpenCL)

add_executable(hands_on_ex2_3 hands_on/ex2_3/main.cpp common/cpp/cl.hpp common/err_code.h common/cpp/util.hpp)
target_link_libraries(hands_on_ex2_3 OpenCL::OpenCL)

add_executable(hands_on_ex4_c hands_on/ex4/main.c common/err_code.h common/c/wtime.c common/c/device_info.c)
target_link_libraries(hands_on_ex4_c OpenCL::OpenCL)

add_executable(hands_on_ex4 hands_on/ex4/main.cpp common/cpp/cl.hpp common/err_code.h common/cpp/util.hpp)
target_link_libraries(hands_on_ex4 OpenCL::OpenCL)

add_executable(hands_on_ex5_c hands_on/ex5/main.c common/err_code.h common/c/wtime.c common/c/device_info.c)
target_link_libraries(hands_on_ex5_c OpenCL::OpenCL)

add_executable(hands_on_ex5 hands_on/ex5/main.cpp common/cpp/cl.hpp common/err_code.h common/cpp/util.hpp)
target_link_libraries(hands_on_ex5 OpenCL::OpenCL)

add_executable(hands_on_ex6_7_8 hands_on/ex6_7_8/main.cpp common/cpp/cl.hpp common/err_code.h common/cpp/util.hpp common/cpp/device_picker.hpp hands_on/ex6_7_8/matrix_lib.cpp hands_on/ex6_7_8/block_mmul.hpp)
target_link_libraries(hands_on_ex6_7_8 OpenCL::OpenCL)

add_executable(hands_on_ex9 hands_on/ex9/main.cpp common/cpp/cl.hpp common/err_code.h common/cpp/util.hpp common/cpp/device_picker.hpp)
target_link_libraries(hands_on_ex9 OpenCL::OpenCL)

