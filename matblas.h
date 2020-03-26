#ifndef MATBLAS_H
#define MATBLAS_H
#include<iostream>
#include<cuda_runtime.h>
#include<cuComplex.h>
using namespace std;
// matblas  Basic Linear Algebra Subprograms
struct cuMat{
    int height;
    int width;
    cuComplex **data;
};
__global__ void cuMatMul(cuMat a, cuMat b, cuMat res, cuComplex alpha); // launch kernel to compute matrix multiplication in parallel
__global__ void cuMatPad(cuMat a, cuMat res, int pad_row, int pad_col); // launch kernel to pad matrix in parallel
__device__ void InitMat(cuMat &mat, int h, int w);                      // Initial matrix with size(h,w) and zero value
__device__ void DestroyMat(cuMat &mat);                                 // Free matrix memory space
__device__ cuMat MulMat(cuMat a, cuMat b, cuComplex alpha);             // return α*op(A)*op(B)  α is scalar, a,b are matrices  
__device__ cuMat PadMat(cuMat a, int pad_row, int pad_col);             // Extend the matrix with size(a.h+2*pad_row, a.w+2*pad_col)
#endif