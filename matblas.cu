#include"matblas.h"

__global__ void cuMatMul(cuMat a, cuMat b, cuMat res, cuComplex alpha) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if( i < a.height && j < b.width){
        res.data[i][j] = make_cuComplex(0.0, 0.0);
        for(int k=0;k<a.width;k++)
        {
            res.data[i][j] = cuCaddf(res.data[i][j],cuCmulf(a.data[i][k],b.data[k][j]));
        }
        res.data[i][j] = cuCmulf(alpha, res.data[i][j]);
    }
    // cuCaddf add two cuComplex; cuCmulf multiply two cuComplex
}

__global__ void  cuMatPad(cuMat a, cuMat res, int pad_row, int pad_col){
    int i = threadIdx.x + blockDim.x * blockIdx.x;  // the ith row
    if(i<res.height){
        if(i<pad_row){
            for(int j = 0;j<pad_col;j++){
                res.data[i][j] = a.data[pad_row - 1 - i][pad_col - 1 -j];
            }
            for(int j = pad_col;j<pad_col+a.width;j++){
                res.data[i][j] = a.data[pad_row - 1 -i][j-pad_col];
            }
            for(int j=pad_col+a.width;j<res.width;j++){
                res.data[i][j] = a.data[pad_col-1-i][2*a.width+pad_col-1-j];
            }
        }else if(i >= pad_row + a.height){
            for(int j = 0;j<pad_col;j++){
                res.data[i][j] = a.data[2*a.height+pad_row-1-i][pad_col - 1 -j];
            }
            for(int j = pad_col;j<pad_col+a.width;j++){
                res.data[i][j] = a.data[pad_row - 1 - i][j-pad_col];
            }
            for(int j=pad_col+a.width;j<res.width;j++){
                res.data[i][j] = a.data[pad_row - 1 -i][2*a.width+pad_col-1-j];
            }
        }else{
            for(int j = 0;j<pad_col;j++){
                res.data[i][j] = a.data[i-pad_row]][pad_col - 1 -j];
            }
            for(int j = pad_col;j<pad_col+a.width;j++){
                res.data[i][j] = a.data[i-pad_row][j-pad_col];
            }
            for(int j=pad_col+a.width;j<res.width;j++){
                res.data[i][j] = a.data[i-pad_row][2*a.width+pad_col-1-j];
            }
        }
    }
}

__device__ void InitMat(cuMat &mat, int h, int w){
    mat.height = h;
    mat.width = w;
    cudaMalloc((void**)&mat.data, sizeof(cuComplex *)*h);
    for(int i=0;i<h;i++)
    {
        cudaMalloc((void**)&mat.data[i], sizeof(cuComplex)*w);
    }
    // Memory does not need to be initialized to ensure speed
}

__device__ void DestroyMat(cuMat &mat){
    for(int i=0;i<mat.height;i++){
        cudaFree(mat.data[i]);
    }
    cudaFree(mat.data);
}

__device__  cuMat  MulMat(cuMat a, cuMat b, cuComplex alpha){
   cuMat res; 
   if(a.width == b.height){
        InitMat(res, a.height, b.width);
        dim3 griddim(3, 3);
        dim3 blockdim(a.height/3 + 1, b.width/3 + 1);
        cuMatMul<<<griddim,blockdim>>>(a, b, res, alpha);    // Compute matrix multiplication in parallel
        cudaDeviceSynchronize();   // parent kernel waits for child kernel 
        return res;
   }else{
        printf("the size of two input Matrix are not match\n");
        InitMat(res, 1, 1);
        return res;
   }
}

__device__ cuMat PadMat(cuMat a, int pad_row, int pad_col){
    cuMat res;
    InitMat(res, a.height + 2*pad_row, a.width + 2*pad_col);
    dim3 blockdim(64);
    dim3 griddim((int)(res.height+63)/64);    // copy by row
    cuMatPad<<<griddim,blockdim>>>(a, res, pad_row, pad_col);
    cudaDeviceSynchronize(); 
    return res;
}