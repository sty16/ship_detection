#include"mat_read.h"
#include"matblas.h"
#include<stdio.h>
#include"device_launch_parameters.h"
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<helper_cuda.h>
#include<helper_functions.h>

//global variable
__device__ __managed__ cuMat imgPad[3]; //device variable managed variable,to which Host is accessible
__device__ __managed__ cuMat img[3];  // the matrix of the origin image

__global__ void PrintData() 
{
    printf("(%f,%f)\n", imgPad[0].data[1015][1015].x, imgPad[0].data[1015][1015].y);
}

__global__ void test() {
    cuFloatComplex *a;
    cudaMalloc((void**)&a, sizeof(cuFloatComplex)*1000);
    // use cuComplex     
    cuMat A, B, C, D;
    cuComplex alpha = make_cuComplex(2.0, 0.0);
    InitMat(A, 6, 6);
    InitMat(B, 6, 6);
    for(int i=0;i<6;i++)
    {
        for(int j=0;j<6;j++)
        {
            A.data[i][j] = make_cuComplex(i,j);
            B.data[i][j] = make_cuComplex(i,j);
        }
    }
    C = MulMat(A, B, alpha);
    __syncthreads();
    // D = PadMat(A, 2, 2);
    __syncthreads();
    DestroyMat(A);
    DestroyMat(B);
    // double *alpha, *beta;
    // alpha = new double(1);
    // beta = new double(0);
    // cublasHandle_t handle;
    // // cublasCreate(&handle);
    // cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3, 3, 2, alpha, d_B, 2, d_B, 2, beta, d_C, 3);
    // printf("%f\n", d_C[1]); 
}


int main(){
    const char *matfile_HH = "./data/imagery_HH.mat";
    const char *param_HH = "imagery_HH";
    const char *matfile_HV = "./data/imagery_HV.mat";
    const char *param_HV = "imagery_HV";
    const char *matfile_VV = "./data/imagery_VV.mat";
    const char *param_VV = "imagery_VV";
    complex<float> *img_HH, *img_HV, *img_VV;
    int h = 1000, w = 1000, N = 15;                               //size of the image data
    img_HH = matToArray(matfile_HH, param_HH);
    img_HV = matToArray(matfile_HV, param_HV);
    img_VV = matToArray(matfile_VV, param_VV);
    for(int i=0;i<3;i++){
        HostInitMat(img[i], h, w);
        cudaMemcpy2D(img[i].meta_data, img[i].pitch, img_HH, sizeof(cuComplex)*w, sizeof(cuComplex)*w, img[i].height, cudaMemcpyHostToDevice);
        imgPad[i] = HostPadMat(img[i], N, N);    // pad to use sliding windows
    }
    PrintData<<<1,1>>>();
    cudaDeviceSynchronize();
    printf("%d\n", img[1].height);
    // test<<<20,20>>>();
    cudaDeviceSynchronize();
    // double *a, *b, *d_B, *d_A, *d_C, *c;
    // cudaMalloc((void**)&d_C, sizeof(double)*3*3);
    // cudaMalloc((void**)&d_B, 6*sizeof(double));
    // cudaMalloc((void**)&d_A, 6*sizeof(double));
    // c = new double[9];
    // b = new double[6];
    // a = new double[6];
    // double alpha = 1, beta = 0;
    // for(int i=0;i<6;i++)
    // {
    //     b[i] = i+1;
    //     a[i] = i+1;
    // }
    // cudaMemcpy(d_A, a, 6*sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, b, 6*sizeof(double), cudaMemcpyHostToDevice);
    // cublasHandle_t handle;
    // stat = cublasCreate(&handle);
    // if (stat != CUBLAS_STATUS_SUCCESS) {
    //     printf ("CUBLAS initialization failed\n");
    //     return EXIT_FAILURE;
    // }
    // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, 2, &alpha, d_A, 3, d_B, 2, &beta, d_C, 3);
    // cudaMemcpy(c, d_C, sizeof(double)*9, cudaMemcpyDeviceToHost);
    // printf("%f\n", c[1]);
    // cublasDestroy(handle);
}