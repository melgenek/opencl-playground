//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix Multiplication driver
//
//  PURPOSE: This is a driver program to test various ways of computing
//           the product:
//
//                C  = A * B
//
//           A and B are set to constant matrices so we
//           can make a quick test of the multiplication.
//
//  USAGE:   The matrices are constant matrices, square and the order is
//           set as a constant, ORDER (see mult.h).
//
//  HISTORY: Written by Tim Mattson, August 2010
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//           Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//           Modified to assume square matrices by Simon McIntosh-Smith, Sep 2014
//
//------------------------------------------------------------------------------

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include "matrix_lib.hpp"
#include "../../common/cpp/util.hpp"
#include "../../common/cpp/device_picker.hpp"
#include "../../common/cpp/cl.hpp"
#include "../../common/err_code.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

const size_t N = 1024;        // A[N][N], B[N][N], C[N][N]
const size_t size = N * N;    // Number of elements in each matrix
const size_t COUNT = 1;       // number of times to do each multiplication

// Exercise 6. Simple
const std::string SIMPLE_MUL = R"(
__kernel void mmul(
   const int N,
   __global float* A,
   __global float* B,
   __global float* C) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < N && j <N) {
        float tmp = 0.0f;
        for (int k = 0; k < N; k++) {
            /* C(i,j) = sum(over k) A(i,k) * B(k,j) */
            tmp += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = tmp;
    }
})";


class ClContext {
public:
    explicit ClContext(size_t deviceIndex) : device(getDeviceList()[deviceIndex]) {}

    [[nodiscard]] cl::CommandQueue createQueue() const {
        return {context, device};
    }

    [[nodiscard]] const cl::Context &getContext() const { return context; }

    [[nodiscard]] const char *getName() const { return deviceName.c_str(); }

private:
    const cl::Device device;
    const std::string deviceName = getDeviceName(device);
    const cl::Context context{device};
};

void multiplyCpuSimple(std::vector<float> &h_A, std::vector<float> &h_B, std::vector<float> &h_C) {
    printf("\n===== Sequential, matrix mul (dot prod), order %zu on host CPU ======\n", N);
    for (int i = 0; i < COUNT; i++) {
        zero_mat(N, h_C);
        util::Timer timer;
        double start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

        seq_mat_mul_sdot(N, h_A, h_B, h_C);

        double run_time = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;
        results(N, h_C, run_time);
    }
}

void multiplyCpuBetterSimple(std::vector<float> &h_A, std::vector<float> &h_B, std::vector<float> &h_C) {
    printf("\n===== Better sequential, matrix mul (dot prod), order %zu on host CPU ======\n", N);
    for (int i = 0; i < COUNT; i++) {
        zero_mat(N, h_C);
        util::Timer timer;
        double start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

        better_seq_mat_mul_sdot(N, h_A, h_B, h_C);

        double run_time = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;
        results(N, h_C, run_time);
    }
}

void multiplyCL(const std::string &name, const std::string &kernelCode, const ClContext &clContext,
                std::vector<float> &h_A,
                std::vector<float> &h_B,
                std::vector<float> &h_C) {
    printf("\n===== OpenCL, matrix mul '%s' on device '%s', order %zu ======\n", name.c_str(), clContext.getName(), N);

    auto queue = clContext.createQueue();
    auto &context = clContext.getContext();

    cl::Program program(context, kernelCode, true);

    auto d_a = cl::Buffer(context, h_A.begin(), h_A.end(), true);
    auto d_b = cl::Buffer(context, h_B.begin(), h_B.end(), true);
    auto d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size);

    auto naive_mmul = cl::KernelFunctor<int, cl::Buffer &, cl::Buffer &, cl::Buffer &>(program, "mmul");

    for (int i = 0; i < COUNT; i++) {
        zero_mat(N, h_C);
        util::Timer timer;
        double start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

        cl::NDRange global(N, N);
        naive_mmul(cl::EnqueueArgs(queue, global),
                   N, d_a, d_b, d_c);

        queue.finish();

        double run_time = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;

        cl::copy(queue, d_c, h_C.begin(), h_C.end());

        results(N, h_C, run_time);
    }
}

int main() {
    std::vector<float> h_A(size);
    std::vector<float> h_B(size);
    std::vector<float> h_C(size);
    initmat(N, h_A, h_B, h_C);

//    multiplyCpuSimple(h_A, h_B, h_C);
    multiplyCpuBetterSimple(h_A, h_B, h_C);

    try {
        const ClContext contextCpu(0);
        multiplyCL("simple", SIMPLE_MUL, contextCpu, h_A, h_B, h_C);

        const ClContext contextIntelGpu(1);
        multiplyCL("simple", SIMPLE_MUL, contextIntelGpu, h_A, h_B, h_C);

        const ClContext contextAmdGpu(2);
        multiplyCL("simple", SIMPLE_MUL, contextAmdGpu, h_A, h_B, h_C);
    } catch (cl::Error &err) {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }

    return EXIT_SUCCESS;
}
