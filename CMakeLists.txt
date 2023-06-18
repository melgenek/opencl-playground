cmake_minimum_required(VERSION 3.7)
project(opencl_playground VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)

add_executable(hands_on_ex1_c hands_on/ex1/main.c common/err_code.h)
target_link_libraries(hands_on_ex1_c OpenCL::OpenCL)

add_executable(hands_on_ex1 hands_on/ex1/main.cpp common/cpp/cl.hpp common/err_code.h)
target_link_libraries(hands_on_ex1 OpenCL::OpenCL)

add_executable(hands_on_ex2_c hands_on/ex2_3/main.c common/err_code.h common/c/wtime.c common/c/device_info.c)
target_link_libraries(hands_on_ex2_c OpenCL::OpenCL)

add_executable(hands_on_ex2 hands_on/ex2_3/main.cpp common/cpp/cl.hpp common/err_code.h common/cpp/util.hpp)
target_link_libraries(hands_on_ex2 OpenCL::OpenCL)
