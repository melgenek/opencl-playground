cmake_minimum_required(VERSION 3.7)
project(opencl_playground VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(opencl_playground main.cpp common/cl.hpp common/err_code.h)
target_compile_features(opencl_playground PRIVATE cxx_auto_type)

find_package(OpenCL REQUIRED)
target_link_libraries(opencl_playground OpenCL::OpenCL)
