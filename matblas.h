#ifndef MATBLAS_H                                                      // matblas  Basic Linear Algebra Subprograms
#define MATBLAS_H
#include<iostream>
#include<cuda_runtime.h>
#include<cuComplex.h>
using namespace std;
#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))
#define BDIMX 32
#define BDIMY BDIMX
typedef unsigned char uint8;
#define THREADSPACE 32768                     // every thread hava global memory 64KB pay attention to that int can express maximum range 2GB

struct cuMat{
    int height;
    int width;
    size_t pitch;                                                      // align with 256B or 512B to speed up memory access
    cuComplex *meta_data;
    cuComplex **data;                                                  // for convenient memory access
};

struct cuImg{
    int height;
    int width;
    size_t pitch;
    uint8 *meta_data;
    uint8 **data;
};

__global__ void cuMatMul(cuMat a, cuMat b, cuMat res, cuComplex alpha); // launch kernel to compute matrix multiplication in parallel
__global__ void cuMatPad(cuMat a, cuMat res, int pad_row, int pad_col); // launch kernel to pad matrix in parallel
__global__ void transposeSmem(cuMat a, cuMat res);                      // launch kernel to tranpose matrix
__global__ void transposeDmem(cuMat a, cuMat res);
__global__ void cuMatHer(cuMat a, cuMat res, cuComplex alpha);
__global__ void cuMatInv(cuMat a, cuMat res, cuComplex det, char *begin, int detsize);  // launch kernel to compute A* adjoint matrix in parallel
__device__ void InitMat(cuMat &mat, int h, int w);                      // Initial matrix with size(h,w) and zero value from GPU
__host__ void HostInitMat(cuMat &mat, int h, int w);                    // Initial matrix with size(h,w) and zero value from CPU
__device__ void DeviceInitMat(cuMat &mat, char *begin, size_t &pointer, int h, int w);     // Initial matrix with size(h,w) on the pre-allocated thread space
__host__ void HostInitImg(cuImg &img,  int h, int w);                   // Initial result img with size(h,w) from CPU
__host__ void HostDestroyImg(cuImg &img);
__device__ void DestroyMat(cuMat &mat);                                 // Free matrix memory space from GPU
__host__ void HostDestroyMat(cuMat &mat);                               // Free matrix memory space from CPU
__device__ void DeviceDestroyMat(cuMat mat, char *begin, size_t &pointer);
__device__ cuMat MulMat(cuMat a, cuMat b, cuComplex alpha);             // return α*op(A)*op(B)  α is scalar, a,b are matrices  
__host__ cuMat HostPadMat(cuMat a, int pad_row, int pad_col);           // Extend the matrix with size(a.h+2*pad_row, a.w+2*pad_col)
__device__ cuMat TransposeMat(cuMat a);                                 // transpose matrix using shared memory
__device__ cuMat DeviceTransMat(cuMat a, char *begin, size_t &pointer);
__device__ cuMat HerMatParal(cuMat mat, char *begin, size_t &pointer, cuComplex alpha);         // compute res = α*mat*mat_H  in which res is a Hermitian matrix
__device__ cuComplex MatDet(cuMat mat, char *begin, size_t &pointer);      // compute matrix determinant
__device__ cuComplex ComputeDet(cuMat mat);                             // only for 3x3 to avoid recursive and speed up
__device__ cuMat MatInv(cuMat mat, char *begin, size_t &pointer);          // compute the inverse Matrix;
__device__ cuMat MatInvParal(cuMat mat, char *begin, size_t &pointer);     // compute the inverse Matrix in parallel
__device__ cuMat HerMat(cuMat a, char *begin, size_t &pointer, cuComplex alpha);  // compute res = α*mat*mat_H  
#endif

//当block中的线程数量大于给定值，则无法执行 32x32
// 动态并行速率大大降低？还是存在什么问题?