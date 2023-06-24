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

// Exercise 6. Simple
const std::string CELL_PER_WORK_ITEM = R"(
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

// Exercise 7. Row per work item
const std::string ROW_PER_WORK_ITEM = R"(
__kernel void mmul(
    const int N,
    __global float* A,
    __global float* B,
    __global float* C) {
    int i = get_global_id(0);

    if (i < N) {
        for (int j = 0; j < N; j++) {
            float tmp = 0.0f;
            for (int k = 0; k < N; k++) {
                tmp += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = tmp;
        }
    }
})";

// Exercise 7. Row per work item with private memory
const std::string ROW_PER_WORK_ITEM_PRIVATE_ROW = R"(
__kernel void mmul(
    const int N,
    __global float* A,
    __global float* B,
    __global float* C) {
    int i = get_global_id(0);

    if (i < N) {
        // float row[N]; // this doesn't compile on Intel UHD Graphics GPU
        float row[1024];
        for (int k = 0; k < N; k++) {
            row[k] = A[i * N + k];
        }

        for (int j = 0; j < N; j++) {
            float tmp = 0.0f;
            for (int k = 0; k < N; k++) {
                tmp += row[k] * B[k * N + j];
            }
            C[i * N + j] = tmp;
        }
    }
})";

// Exercise 8. Row per work item with private memory for row and local memory for column.
const std::string ROW_PER_WORK_ITEM_PRIVATE_ROW_LOCAL_COLUMN = R"(
__kernel void mmul(
    const int N,
    __global float* A,
    __global float* B,
    __global float* C,
    __local float* column) {
    int i = get_global_id(0);
    int iloc = get_local_id(0);
    int nloc = get_local_size(0);

    if (i < N) {
        // float row[N]; // this doesn't compile on Intel UHD Graphics GPU
        float row[1024];
        for (int k = 0; k < N; k++) {
            row[k] = A[i * N + k];
        }

        for (int j = 0; j < N; j++) {
            for (int k=iloc; k < N; k += nloc) {
                column[k] = B[k * N + j];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            float tmp = 0.0f;
            for (int k = 0; k < N; k++) {
                tmp += row[k] * column[k];
            }
            C[i * N + j] = tmp;

            barrier(CLK_LOCAL_MEM_FENCE);
        }
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
    printf("Sequential, matrix mul (dot prod), order %zu on host CPU,\t", N);
    zero_mat(N, h_C);
    util::Timer timer;
    double start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

    seq_mat_mul_sdot(N, h_A, h_B, h_C);

    double run_time = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;
    results(N, h_C, run_time);
    printf("\n");
}

void multiplyCpuBetterSimple(std::vector<float> &h_A, std::vector<float> &h_B, std::vector<float> &h_C) {
    printf("Better sequential, matrix mul (dot prod), order %zu on host CPU,\t", N);
    zero_mat(N, h_C);
    util::Timer timer;
    double start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

    better_seq_mat_mul_sdot(N, h_A, h_B, h_C);

    double run_time = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;
    results(N, h_C, run_time);
    printf("\n");
}

void multiplyCL(const ClContext &clContext,
                const std::string &name,
                const std::string &kernelCode,
                const std::function<cl::EnqueueArgs(cl::CommandQueue &)> &createArgs,
                std::vector<float> &h_A,
                std::vector<float> &h_B,
                std::vector<float> &h_C) {
    printf("OpenCL, matrix mul '%s', order %zu,\t",
           name.c_str(),
           N);

    auto queue = clContext.createQueue();
    auto &context = clContext.getContext();

    cl::Program program(context, kernelCode, true);

    auto d_a = cl::Buffer(context, h_A.begin(), h_A.end(), true);
    auto d_b = cl::Buffer(context, h_B.begin(), h_B.end(), true);
    auto d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size);

    auto naive_mmul = cl::KernelFunctor<int, cl::Buffer &, cl::Buffer &, cl::Buffer &>(program, "mmul");

    zero_mat(N, h_C);
    util::Timer timer;
    double start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

    naive_mmul(createArgs(queue), N, d_a, d_b, d_c);

    queue.finish();

    double run_time = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;

    cl::copy(queue, d_c, h_C.begin(), h_C.end());

    results(N, h_C, run_time);
}


void multiplyCLWithLocalColumn(const ClContext &clContext,
                               const std::string &name,
                               const std::function<cl::EnqueueArgs(cl::CommandQueue &)> &createArgs,
                               std::vector<float> &h_A,
                               std::vector<float> &h_B,
                               std::vector<float> &h_C) {
    printf("OpenCL, matrix mul '%s', order %zu,\t",
           name.c_str(),
           N);

    auto queue = clContext.createQueue();
    auto &context = clContext.getContext();

    cl::Program program(context, ROW_PER_WORK_ITEM_PRIVATE_ROW_LOCAL_COLUMN, true);

    auto d_a = cl::Buffer(context, h_A.begin(), h_A.end(), true);
    auto d_b = cl::Buffer(context, h_B.begin(), h_B.end(), true);
    auto d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size);

    auto naive_mmul = cl::KernelFunctor<int, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::LocalSpaceArg>(program,
                                                                                                          "mmul");

    zero_mat(N, h_C);
    util::Timer timer;
    double start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

    naive_mmul(createArgs(queue), N, d_a, d_b, d_c, cl::Local(sizeof(float) * N));

    queue.finish();

    double run_time = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;

    cl::copy(queue, d_c, h_C.begin(), h_C.end());

    results(N, h_C, run_time);
}

void runForDevice(size_t deviceIndex,
                  std::vector<float> &h_A,
                  std::vector<float> &h_B,
                  std::vector<float> &h_C) {
    const ClContext clContext(deviceIndex);

    printf("===== Device '%s' start =====\n", clContext.getName());
    multiplyCL(clContext, "C(i,j) per work item", CELL_PER_WORK_ITEM, [](auto queue) {
        return cl::EnqueueArgs(queue, cl::NDRange(N, N));
    }, h_A, h_B, h_C);
    multiplyCL(clContext, "C row per work item, 16 units", ROW_PER_WORK_ITEM, [](auto queue) {
        return cl::EnqueueArgs(queue, cl::NDRange(N), cl::NDRange(N / 16));
    }, h_A, h_B, h_C);
    multiplyCL(clContext, "C row per work item, any units", ROW_PER_WORK_ITEM, [](auto queue) {
        return cl::EnqueueArgs(queue, cl::NDRange(N));
    }, h_A, h_B, h_C);
    if (deviceIndex != 0) { // Intel CPU gives CL_INVALID_WORK_GROUP_SIZE
        multiplyCL(clContext, "C row per work item private memory, 16 units",
                   ROW_PER_WORK_ITEM_PRIVATE_ROW,
                   [](auto queue) {
                       return cl::EnqueueArgs(queue, cl::NDRange(N), cl::NDRange(N / 16));
                   }, h_A, h_B, h_C);
    }
    multiplyCL(clContext, "C row per work item private memory, any units", ROW_PER_WORK_ITEM_PRIVATE_ROW,
               [](auto queue) {
                   return cl::EnqueueArgs(queue, cl::NDRange(N));
               }, h_A, h_B, h_C);

    if (deviceIndex != 0) { // Intel CPU gives CL_INVALID_WORK_GROUP_SIZE
        multiplyCLWithLocalColumn(clContext, "C row per work item, 16 units", [](auto queue) {
            return cl::EnqueueArgs(queue, cl::NDRange(N), cl::NDRange(N / 16));
        }, h_A, h_B, h_C);
    }
    multiplyCLWithLocalColumn(clContext, "C row per work item, any units", [](auto queue) {
        return cl::EnqueueArgs(queue, cl::NDRange(N));
    }, h_A, h_B, h_C);
    printf("===== Device '%s' done =====\n\n", clContext.getName());
}

int main() {
    std::vector<float> h_A(size);
    std::vector<float> h_B(size);
    std::vector<float> h_C(size);
    initmat(N, h_A, h_B, h_C);

    multiplyCpuSimple(h_A, h_B, h_C);
    multiplyCpuBetterSimple(h_A, h_B, h_C);

    try {
        for (int i = 0; i <= 2; i++) {
            runForDevice(i, h_A, h_B, h_C);
        }
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
