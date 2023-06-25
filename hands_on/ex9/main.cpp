/*

This program will numerically compute the integral of

                  4/(1+x*x)

from 0 to 1.  The value of this integral is pi -- which
is great since it gives us an easy way to check the answer.

*/

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include "../../common/cpp/util.hpp"
#include "../../common/cpp/device_picker.hpp"
#include "../../common/cpp/cl.hpp"
#include "../../common/err_code.h"

#include <cstdio>

const unsigned long num_steps = 100000000L;
const double step = 1.0 / (double) num_steps;

// Exercise 9. Simple PI calculation
// Floating number type is `float`, because Intel UHD doesn't have an OpenCL extension for double precision `cl_khr_fp64`.
const std::string SIMPLE_PI = R"(
__kernel void pi(
    const unsigned long num_steps,
    const float step,
    __local float* worker_group_results,
    __global float* all_results) {
    int local_id = get_local_id(0);
    int global_size = get_global_size(0);
    int group_id = get_group_id(0);
    int global_id = get_global_id(0);

    int steps_per_work_item = num_steps / global_size;

    float sum = 0.0f;
    for(int i = global_id * steps_per_work_item; i < (global_id+1)*steps_per_work_item; i++) {
        float x = (i - 0.5f) * step;
        sum += 4.0f / (1.0f + x * x);
    }
    worker_group_results[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        float group_sum = 0.0f;
        for (size_t i = 0; i < get_local_size(0); i++) {
            group_sum += worker_group_results[i];
        }
        all_results[group_id] = group_sum*step;
    }
})";

// Appendix A exercise. float4
const std::string FLOAT4_PI = R"(
__kernel void pi(
    const unsigned long num_steps,
    const float step,
    __local float* worker_group_results,
    __global float* all_results) {
    int local_id = get_local_id(0);
    int global_size = get_global_size(0);
    int group_id = get_group_id(0);
    int global_id = get_global_id(0);

    int steps_per_work_item = num_steps / global_size;

    float4 sum_vec = {0.0, 0.0, 0.0, 0.0};
    float4 offset = {0.5f, 1.5f, 2.5f, 3.5f};

    for(int i = global_id * steps_per_work_item; i < (global_id+1)*steps_per_work_item; i+=4) {
        float4 x = ((float4)i + offset) * step;
        sum_vec += 4.0f / (1.0f + x * x);
    }
    worker_group_results[local_id] = sum_vec.s0 + sum_vec.s1 + sum_vec.s2 + sum_vec.s3;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        float group_sum = 0.0f;
        for (size_t i = 0; i < get_local_size(0); i++) {
            group_sum += worker_group_results[i];
        }
        all_results[group_id] = group_sum*step;
    }
})";

// Appendix A exercise. float8
const std::string FLOAT8_PI = R"(
__kernel void pi(
    const unsigned long num_steps,
    const float step,
    __local float* worker_group_results,
    __global float* all_results) {
    int local_id = get_local_id(0);
    int global_size = get_global_size(0);
    int group_id = get_group_id(0);
    int global_id = get_global_id(0);

    int steps_per_work_item = num_steps / global_size;

    float8 sum_vec = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float8 offset = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f};

    for(int i = global_id * steps_per_work_item; i < (global_id+1)*steps_per_work_item; i+=8) {
        float8 x = ((float8)i + offset) * step;
        sum_vec += 4.0f / (1.0f + x * x);
    }
    worker_group_results[local_id] = sum_vec.s0 + sum_vec.s1 + sum_vec.s2 + sum_vec.s3 + sum_vec.s4 + sum_vec.s5 + sum_vec.s6 + sum_vec.s7;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        float group_sum = 0.0f;
        for (size_t i = 0; i < get_local_size(0); i++) {
            group_sum += worker_group_results[i];
        }
        all_results[group_id] = group_sum*step;
    }
})";

void find_pi_sequentially() {
    util::Timer timer;
    double sum = 0.0;
    for (int i = 0; i < num_steps; i++) {
        double x = (i - 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    float pi = step * sum;

    double run_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

    printf("pi with %ld steps is %lf in %lf seconds\n", num_steps, pi, run_time);
}

void find_pi_cl(const std::string &kernelCode, const std::string &name_info) {
    for (const auto &device: getDeviceList()) {
        const std::string deviceName = getDeviceName(device);
        const cl::Context context(device);
        cl::CommandQueue queue(context, device);

        cl::Program program(context, kernelCode);
        try {
            program.build();
        }
        catch (cl::Error &err) {
            cl_int buildErr = CL_SUCCESS;
            auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
            for (auto &pair: buildInfo) {
                std::cerr << pair.second << std::endl << std::endl;
            }
            throw err;
        }
        auto pi_kernel = cl::KernelFunctor<unsigned long, float, cl::LocalSpaceArg, cl::Buffer>(
                program, "pi");

        size_t work_group_size = pi_kernel.getKernel().getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(
                device);
        uint32_t compute_units = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        size_t global_size = work_group_size * compute_units;

        util::Timer timer;
        std::vector<float> h_worker_group_sums(compute_units);
        auto d_worker_group_sums = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * compute_units);

        cl::LocalSpaceArg local_mem_size = cl::Local(sizeof(float) * work_group_size);
        pi_kernel(
                cl::EnqueueArgs(
                        queue,
                        cl::NDRange(global_size),
                        cl::NDRange(work_group_size)),
                num_steps,
                static_cast<float>(step),
                local_mem_size,
                d_worker_group_sums);
        queue.finish();
        cl::copy(queue, d_worker_group_sums, h_worker_group_sums.begin(), h_worker_group_sums.end());

        float pi = 0.0;
        for (float v: h_worker_group_sums) {
            pi += v;
        }

        double run_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
        printf("pi with %ld steps is %lf in %lf seconds. %s. WG size: %zu, CU: %u. Device: %s\n", num_steps, pi,
               run_time,
               name_info.c_str(), work_group_size, compute_units, deviceName.c_str());
    }
}

int main() {
    find_pi_sequentially();

    try {
        find_pi_cl(SIMPLE_PI, "simple");
        find_pi_cl(FLOAT4_PI, "float4");
        find_pi_cl(FLOAT8_PI, "float8");
    } catch (cl::Error &err) {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }
}

