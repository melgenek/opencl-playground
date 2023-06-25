//------------------------------------------------------------------------------
//
// Name:       vadd_cpp.cpp
// 
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
//                   c = a + b
//             
//------------------------------------------------------------------------------

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120


#include "../../common/cpp/cl.hpp"
#include "../../common/cpp/util.hpp"
#include "../../common/err_code.h"

#include <vector>
#include <cstdio>
#include <cstdlib>

#include <iostream>

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

//------------------------------------------------------------------------------

const float TOL = 0.001;   // tolerance used in floating point comparisons
const size_t LENGTH = 1024; // length of vectors a, b, and c

const std::string ADD_KERNEL = R"(
__kernel void vadd(
   __global float* a,
   __global float* b,
   __global float* c,
   __global float* d,
   const unsigned int count)
{
   int i = get_global_id(0);
   if(i < count)  {
       d[i] = a[i] + b[i] + c[i];
   }
})";

int main() {
    std::vector<float> h_a(LENGTH);                // a vector 
    std::vector<float> h_b(LENGTH);                // b vector 	
    std::vector<float> h_c(LENGTH);                // c vector
    std::vector<float> h_d(LENGTH, 0xdeadbeef);    // c = a + b, from compute device

    cl::Buffer d_a;                        // device memory used for the input  a vector
    cl::Buffer d_b;                        // device memory used for the input  b vector
    cl::Buffer d_c;                       // device memory used for the output c vector
    cl::Buffer d_d;                       // device memory used for the output d vector

    // Fill vectors a and b with random float values
    for (int i = 0; i < LENGTH; i++) {
        h_a[i] = rand() / (float) RAND_MAX;
        h_b[i] = rand() / (float) RAND_MAX;
        h_c[i] = rand() / (float) RAND_MAX;
    }

    try {
        // Create a context
        cl::Context context(DEVICE);

        // Load in kernel source, creating a program object for the context

        cl::Program program(context, ADD_KERNEL, true);

        // Get the command queue
        cl::CommandQueue queue(context);

        // Create the kernel functor
        auto vadd = cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, int>(program, "vadd");

        d_a = cl::Buffer(context, begin(h_a), end(h_a), true);
        d_b = cl::Buffer(context, begin(h_b), end(h_b), true);
        d_c = cl::Buffer(context, begin(h_c), end(h_c), true);
        d_d = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

        util::Timer timer;

        vadd(cl::EnqueueArgs(queue, cl::NDRange(LENGTH)),
             d_a,
             d_b,
             d_c,
             d_d,
             LENGTH);

        queue.finish();

        printf("The kernels ran in %llu ms\n", timer.getTimeMilliseconds());

        cl::copy(queue, d_d, begin(h_d), end(h_d));

        // Test the results
        int correct = 0;
        float tmp;
        for (int i = 0; i < LENGTH; i++) {
            tmp = h_a[i] + h_b[i] + h_c[i]; // expected value for d_c[i]
            tmp -= h_d[i];                      // compute errors
            if (tmp * tmp < TOL * TOL) {      // correct if square deviation is less
                correct++;                         //  than tolerance squared
            } else {
                printf(
                        " tmp %f h_a %f h_b %f  h_c %f h_d %f \n",
                        tmp,
                        h_a[i],
                        h_b[i],
                        h_c[i],
                        h_d[i]);
            }
        }

        // summarize results
        printf("vector add to find D = A+B+C:  %d out of %zu results were correct.\n", correct, LENGTH);
    }
    catch (cl::Error &err) {
        std::cout << "Exception\n";
        std::cerr
                << "ERROR: "
                << err.what()
                << "("
                << err_code(err.err())
                << ")"
                << std::endl;
    }
}
