#include"mat_read.h"
#include"matblas.h"
#include<stdio.h>
#include<math.h>
#include"device_launch_parameters.h"
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<helper_cuda.h>
#include<helper_functions.h>

//global variable
__device__ __managed__ cuMat imgPad[3];          //device variable managed variable,to which Host is accessible
__device__ __managed__ cuMat img[3];             // the matrix of the origin image
__device__ __managed__ char *MemPool;            // the global memory that thread have  内存池
__device__ __managed__ cuImg result;             // the  result of the pwf image


__global__ void CopyData(cuMat out, int row, int col, int pad_row, int pad_col){
    int i = threadIdx.x;
    int j = threadIdx.y;
    if(i < 2*pad_row && j < 2*pad_col)
    {
        int col_out = i*2*pad_col + j;
        for(int k = 0;k < 3;k++)
        {
            out.meta_data[INDEX(k, col_out, out.width)] =  imgPad[k].data[row+i][col+j];
        }
    }
}
__global__ void f_PWF(int pad_row, int pad_col, int row) 
{
    int pointer = 0, c_row, c_col, t_row, t_col, N=900;                // set the pointer 0 && set the sliding window size 30
    cuComplex  sigmac_inv[9], s[9], sigmac_det;
    c_row = threadIdx.x;
    c_col = threadIdx.y;
    t_row = row;
    t_col = INDEX(c_row, c_col, blockDim.y);
    char *ThreadMemPool = (char *)MemPool + INDEX(c_row, c_col, blockDim.y)*THREADSPACE;
    cuMat clutter, sigma_c, test;
    if(t_col < img[1].width)
    {
        DeviceInitMat(clutter, ThreadMemPool, pointer, 3, N);
        dim3 gdim(1, 1);
        dim3 bdim(32, 32);
        CopyData<<<gdim, bdim>>>(clutter, t_row, t_col, pad_row, pad_col);    // copy global data to the thread clutter memory
        cudaDeviceSynchronize();
        cuComplex alpha = make_cuComplex(1.0/N, 0);
        sigma_c = HerMat(clutter, ThreadMemPool, pointer, alpha);
        for(int i=0;i < 3;i++)
        {
            for(int j=0;j<3;j++)
            {
                s[INDEX(i, j, 3)] = sigma_c.meta_data[INDEX(i, j, sigma_c.width)]; 
            }
        }
    }
    if(t_row==0&&t_col == 1){
        cuMat temp1;
        DeviceInitMat(temp1, ThreadMemPool, pointer, 3, 3);
        for(int i=0;i<3;i++){
            for(int j = 0;j<3;j++){
                temp1.meta_data[INDEX(i, j, temp1.width)] = make_cuComplex(i*3+j+1,0);
            }
        }
        temp1.meta_data[INDEX(0,2,temp1.width)] = make_cuComplex(1,0);
        cuComplex det = MatDet(temp1, ThreadMemPool, pointer);
        printf("%f,%f\n", det.x, det.y);
        DeviceDestroyMat(temp1, ThreadMemPool, pointer);
        DeviceInitMat(test, ThreadMemPool, pointer, 30, 40);
        printf("%d\n", pointer);
        DeviceDestroyMat(test, ThreadMemPool, pointer);
        printf("%d\n", pointer);
        printf("%f\n", sigma_c.meta_data[INDEX(0, 0, sigma_c.width)].x);
        printf("(%f,%f)", clutter.meta_data[INDEX(0,0,clutter.width)].x, clutter.meta_data[INDEX(0,0,clutter.width)].y);
    }
    // double *temp;
    // cudaError_t error_t = cudaMalloc((void **)temp, sizeof(double)*1000000); //分配的不是全局内存空间
    // switch( error_t )
    // {
    //   case cudaSuccess: printf("cudaSuccess\n");break;
    //   case cudaErrorMemoryAllocation:printf("cudaErrorMemoryAllocation\n");break;
    //   default:printf("default: %d \n",error_t );break;
    // }
    // cuComplex alpha = make_cuComplex(1/N, 0);
    // sigma_c = MulMat(clutter, clutter_t, alpha);
    // if(c_col == 0){
    //     printf("%f\n", sigma_c.data[0][0].x);
    // }
    // // B = TransposeMat(A);
    // printf("%d", B.height);
    // for(int i = 0;i<1;i++){
    //     for(int j=0;j<M;j++){
    //         printf("(%f,%f) " , B.data[i][j].x, B.data[i][j].y);
    //     }
    //     printf("\n");
    // }
    // printf("%d", A.height);
}

__host__ float  get_T_PWF(int num, float P, int maxIter=500, double ep = 1e-5, double x0 = 1)
{
    double x, fx, dfx;
    double Pfa = (double)P;
    for (int i=0; i< maxIter;i++)
    {
        fx = exp(-x0) * (1 + x0 + pow(x0, 2)/ 2) - Pfa;
        dfx = -exp(-x0) * (pow(x0, 2)/ 2);
        x = x0 - fx / dfx;
        if ((fx < ep) and (fx > -ep)) break; 
        x0 = x;
    }
    return (float)x;
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
    float Pfa = 0.001;
    float T = get_T_PWF(3, Pfa);
    // printf("%f", T);
    img_HH = matToArray(matfile_HH, param_HH);
    img_HV = matToArray(matfile_HV, param_HV);
    img_VV = matToArray(matfile_VV, param_VV);
    for(int i=0;i<3;i++){
        HostInitMat(img[i], h, w);
        cudaMemcpy2D(img[i].meta_data, img[i].pitch, img_HH, sizeof(cuComplex)*w, sizeof(cuComplex)*w, img[i].height, cudaMemcpyHostToDevice);
        imgPad[i] = HostPadMat(img[i], N, N);    // pad to use sliding windows
    }
    HostInitImg(result, h, w);
    dim3 griddim(1,1);
    dim3 blockdim(32,32);
    int size = THREADSPACE*blockdim.x*blockdim.y*griddim.x*griddim.y;
    printf("%d", size);
    cudaError_t error_t = cudaMalloc((void **)&MemPool, sizeof(char)*size); //分配的不是全局内存空间
    switch( error_t )
    {
      case cudaSuccess: printf("cudaSuccess\n");break;
      case cudaErrorMemoryAllocation:printf("cudaErrorMemoryAllocation\n");break;
      default:printf("default: %d \n",error_t );break;
    }
    int row=0;
    f_PWF<<<griddim,blockdim>>>(N, N, row);
    cudaDeviceSynchronize();
    printf("%d\n", img[1].height);
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