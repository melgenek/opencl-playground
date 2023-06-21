//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix library for the multiplication driver
//
//  PURPOSE: This is a simple set of functions to manipulate
//           matrices used with the multiplcation driver.
//
//  USAGE:   The matrices are square and the order is
//           set as a defined constant, ORDER.
//
//  HISTORY: Written by Tim Mattson, August 2010
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//           Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//           Modified to assume square matrices by Simon McIntosh-Smith, Sep 2014
//
//------------------------------------------------------------------------------

#include "matrix_lib.hpp"

const float AVAL = 3.0;    // A elements are constant and equal to AVAL
const float BVAL = 5.0;    // B elements are constant and equal to BVAL
const float TOL = 0.001;   // tolerance used in floating point comparisons
//------------------------------------------------------------------------------
//
//  Function to compute the matrix product (sequential algorithm, dot prod)
//
//------------------------------------------------------------------------------



void seq_mat_mul_sdot(int N, std::vector<float> &A, std::vector<float> &B, std::vector<float> &C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float tmp = 0.0f;
            for (int k = 0; k < N; k++) {
                /* C(i,j) = sum(over k) A(i,k) * B(k,j) */
                tmp += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = tmp;
        }
    }
}

// 1 2
// 3 4

// 5 6
// 7 8

// 1*5+2*7 1*6+2*8
// 3*5+4*7 3*6+4*8

void better_seq_mat_mul_sdot(int N, std::vector<float> &A, std::vector<float> &B, std::vector<float> &C) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

//------------------------------------------------------------------------------
//
//  Function to initialize the input matrices A and B
//
//------------------------------------------------------------------------------
void initmat(int N, std::vector<float> &A, std::vector<float> &B, std::vector<float> &C) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i * N + j] = AVAL;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            B[i * N + j] = BVAL;

    zero_mat(N, C);
}

//------------------------------------------------------------------------------
//
//  Function to set a matrix to zero
//
//------------------------------------------------------------------------------
void zero_mat(int N, std::vector<float> &C) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i * N + j] = 0.0f;
}

//------------------------------------------------------------------------------
//
//  Function to fill Btrans(N,N) with transpose of B(N,N)
//
//------------------------------------------------------------------------------
void trans(int N, std::vector<float> &B, std::vector<float> &Btrans) {
    int i, j;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            Btrans[j * N + i] = B[i * N + j];
}

//------------------------------------------------------------------------------
//
//  Function to compute errors of the product matrix
//
//------------------------------------------------------------------------------
float error(int N, std::vector<float> &C) {
    int i, j;
    float cval, errsq, err;
    cval = (float) N * AVAL * BVAL;
    errsq = 0.0f;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            err = C[i * N + j] - cval;
            errsq += err * err;
        }
    }
    return errsq;
}

//------------------------------------------------------------------------------
//
//  Function to analyze and output results
//
//------------------------------------------------------------------------------
void results(int N, std::vector<float> &C, double run_time) {
    float mflops = 2.0 * N * N * N / (1000000.0f * run_time);
    printf(" %.4f seconds at %.1f MFLOPS \n", run_time, mflops);
    float errsq = error(N, C);
    if (std::isnan(errsq) || errsq > TOL)
        printf("\n Errors in multiplication: %f\n", errsq);
}

