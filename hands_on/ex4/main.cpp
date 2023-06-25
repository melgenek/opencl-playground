//------------------------------------------------------------------------------
//
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
//                   c = a + b
//
//------------------------------------------------------------------------------

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120


#include "../common/cpp/cl.hpp"
#include "../common/cpp/util.hpp"
#include "../common/err_code.h"

#include <vector>
#include <cstdio>
#include <cstdlib>

#include <iostream>

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_GPU
#endif

//------------------------------------------------------------------------------

const float TOL = 0.001;   // tolerance used in floating point comparisons
const size_t LENGTH = 1024 * 1024 * 1024 / 16;

const std::string ADD_KERNEL = R"(
__kernel void vadd(
   __global float* a,
   __global float* b,
   __global float* c,
   const unsigned int count)
{
   int i = get_global_id(0);
   if(i < count)  {
       c[i] = a[i] + b[i];
   }
})";

int main() {
    std::vector<float> h_a(LENGTH);                // a vector
    std::vector<float> h_b(LENGTH);                // b vector
    std::vector<float> h_c(LENGTH, 0xdeadbeef);    // c = a + b, from compute device
    std::vector<float> h_d(LENGTH, 0xdeadbeef);
    std::vector<float> h_e(LENGTH);
    std::vector<float> h_f(LENGTH, 0xdeadbeef);
    std::vector<float> h_g(LENGTH);

    cl::Buffer d_a;                        // device memory used for the input  a vector
    cl::Buffer d_b;                        // device memory used for the input  b vector
    cl::Buffer d_c;                       // device memory used for the output c vector
    cl::Buffer d_e;
    cl::Buffer d_d;
    cl::Buffer d_f;
    cl::Buffer d_g;

    // Fill vectors a and b with random float values
    for (int i = 0; i < LENGTH; i++) {
        h_a[i] = rand() / (float) RAND_MAX;
        h_b[i] = rand() / (float) RAND_MAX;
        h_e[i] = rand() / (float) RAND_MAX;
        h_g[i] = rand() / (float) RAND_MAX;
    }

    try {
        // Create a context
        cl::Context context(DEVICE);

        // Load in kernel source, creating a program object for the context

        cl::Program program(context, ADD_KERNEL, true);

        // Get the command queue
        cl::CommandQueue queue(context);

        // Create the kernel functor
        auto vadd = cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &, int>(program, "vadd");

        d_a = cl::Buffer(context, begin(h_a), end(h_a), true);
        d_b = cl::Buffer(context, begin(h_b), end(h_b), true);
        d_c = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * LENGTH);
        d_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * LENGTH);
        d_e = cl::Buffer(context, begin(h_e), end(h_e), true);
        d_f = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);
        d_g = cl::Buffer(context, begin(h_g), end(h_g), true);

        util::Timer timer;

        vadd(cl::EnqueueArgs(queue, cl::NDRange(LENGTH)),
             d_a,
             d_b,
             d_c,
             LENGTH);

        vadd(cl::EnqueueArgs(queue, cl::NDRange(LENGTH)),
             d_c,
             d_e,
             d_d,
             LENGTH);

        vadd(cl::EnqueueArgs(queue, cl::NDRange(LENGTH)),
             d_d,
             d_g,
             d_f,
             LENGTH);

        queue.finish();

        printf("The kernels ran in %llu ms\n", timer.getTimeMilliseconds());

        cl::copy(queue, d_f, begin(h_f), end(h_f));

        // Test the results
        int correct = 0;
        float tmp;
        for (int i = 0; i < LENGTH; i++) {
            tmp = h_a[i] + h_b[i] + h_e[i] + h_g[i];
            tmp -= h_f[i];                      // compute errors
            if (tmp * tmp < TOL * TOL) {      // correct if square deviation is less
                correct++;                         //  than tolerance squared
            } else {
                printf(
                        " tmp %f h_a %f h_b %f h_e %f h_g %f = h_f %f \n",
                        tmp,
                        h_a[i],
                        h_b[i],
                        h_e[i],
                        h_g[i],
                        h_f[i]);
            }
        }

        // summarize results
        printf("vector add to find F = A+B+E+G:  %d out of %zu results were correct.\n", correct, LENGTH);
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
