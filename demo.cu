#include"mat_read.h"
#include"matblas.h"
#include<stdio.h>
#include<math.h>
#include"device_launch_parameters.h"
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<opencv2/opencv.hpp>
#include<helper_cuda.h>
#include<helper_functions.h>

using namespace cv;

//global variable
__device__ __managed__ cuMat imgPad[3];          //device variable managed variable,to which Host is accessible
__device__ __managed__ cuMat img[3];             // the matrix of the origin image
__device__ __managed__ char *MemPool;            // the global memory that thread have  内存池
__device__ __managed__ cuImg resImg;             // the  result of the pwf image


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

__device__ void copyToClutter(cuMat clutter, int row, int col, int pad_row, int pad_col)
{
    // pad_row = 7 pad_col = 7
    for(int k=0;k<3;k++)
    {
        for(int i=0;i<2*pad_row+1;i++)
        {
            for(int j = 0;j<2*pad_col+1;j++)
            {
                int temp = i*(2*pad_col+1) + j;
                if(i==(row+pad_row) && j==(col+pad_col))
                {
                    continue;
                }
                clutter.meta_data[INDEX(k, temp, clutter.width)] = imgPad[k].meta_data[INDEX(row+i, col+j, imgPad[k].pitch/sizeof(cuComplex))];
            }
        }
    }
}
// 多线程中复制使用多线程会显著降低性能，因为此时的线程数量过多

__global__ void f_PWF(int pad_row, int pad_col, int num, float T) 
{
    size_t pointer = 0;
    int c_row, t_row, t_col, N=(2*pad_row+1)*(2*pad_col+1) - 1;                // set the pointer 0 && set the sliding window size 30
    cuComplex  data[3], temp[3], result;
    float res;
    c_row = blockIdx.x*gridDim.y + blockIdx.y;
    t_row = c_row + num*gridDim.x*gridDim.y;
    t_col = INDEX(threadIdx.x, threadIdx.y, blockDim.y);
    if(t_row < img[0].height && t_col < img[0].width)
    {
        char *ThreadMemPool = (char *)MemPool + (size_t)INDEX(c_row, t_col, img[0].width)*THREADSPACE;   // 找到线程自己的全局内存池位置
        cuMat clutter, sigma_c, sigma_inv;
        DeviceInitMat(clutter, ThreadMemPool, pointer, 3, N);
        copyToClutter(clutter, t_row, t_col, pad_row, pad_col);
        cuComplex alpha = make_cuComplex(1.0/N, 0);
        sigma_c = HerMat(clutter, ThreadMemPool, pointer, alpha);
        sigma_inv = MatInv(sigma_c, ThreadMemPool, pointer);
        for(int i=0;i < 3;i++)
        {
            data[i] =  img[i].meta_data[INDEX(t_row, t_col, img[i].pitch/sizeof(cuComplex))]; // 注意mallocpitch要行对其访问
        }
        for(int i=0;i<3;i++)
        {
            temp[i] = make_cuComplex(0, 0);
            for(int j=0;j<3;j++)
            {
                temp[i] = cuCaddf(temp[i], cuCmulf(sigma_inv.meta_data[INDEX(i, j, sigma_inv.width)], data[j]));
            }
        }
        result = make_cuComplex(0, 0);
        for(int i=0;i<3;i++)
        {
            result = cuCaddf(result, cuCmulf(cuConjf(data[i]), temp[i]));
        }
        res = cuCabsf(result);
        resImg.data[t_row][t_col] = res>T?255:0;
        // printf("%lu\n", pointer);
        // printf("%f\n", res);
        if(t_col == 0 && t_row == 0)
        {
            // cuComplex det = MatDet(sigma_c, ThreadMemPool, pointer);
            // printf("%f, %f\n", det.x, det.y);
            // printf("%f,%f\n", sigma_c.meta_data[0].x, sigma_c.meta_data[0].y);
            // printf("%f\n", res);
            // printf("%f,%f\n", data[2].x, data[2].y);
            // for(int i=0;i<clutter.width;i++)
            // {
            //     printf("%f,%f\n", clutter.meta_data[i].x, clutter.meta_data[i].y);
            // }
            // printf("%d\n", t_row);
        }
    }

        // printf("%f\n", res);
        // cuMat temp1, temp1_inv;
        // DeviceInitMat(temp1, ThreadMemPool, pointer, 4, 4);
        // for(int i=0;i<4;i++){
        //     for(int j = 0;j<4;j++){
        //         temp1.meta_data[INDEX(i, j, temp1.width)] = make_cuComplex(i*4+j+1,0);
        //     }
        // }
        // temp1.meta_data[INDEX(0,2,temp1.width)] = make_cuComplex(1,0);
        // temp1.meta_data[INDEX(1,2,temp1.width)] = make_cuComplex(0,0);
        // temp1.meta_data[INDEX(2,1,temp1.width)] = make_cuComplex(1,0);
        // cuComplex det = MatDet(temp1, ThreadMemPool, pointer);
        // printf("%f,%f\n",det.x, det.y);
        // temp1_inv = MatInvParal(temp1, ThreadMemPool, pointer);
        // for(int i=0;i<4;i++){
        //     for(int j = 0;j<4;j++){
        //         if(t_col == 1){
        //             printf("(%f,%f)", temp1_inv.meta_data[INDEX(i, j, temp1_inv.width)].x, temp1_inv.meta_data[INDEX(i, j, temp1_inv.width)].y);
        //         }
        //     }
        //     if(t_col == 1){
        //         printf("\n");
        //     }
        // }
        // printf("ok");
    // }
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

int main(){
    const char *matfile_HH = "./data/imagery_HH.mat";
    const char *param_HH = "imagery_HH";
    const char *matfile_HV = "./data/imagery_HV.mat";
    const char *param_HV = "imagery_HV";
    const char *matfile_VV = "./data/imagery_VV.mat";
    const char *param_VV = "imagery_VV";
    complex<float> *img_mat[3];
    int h = 1000, w = 1000, N = 7;                               //size of the image data
    float Pfa = 0.001; char *resImg_host;
    float T = get_T_PWF(3, Pfa);
    resImg_host = new char[h*w];
    // printf("%f", T);
    img_mat[0] = matToArray(matfile_HH, param_HH);
    img_mat[1] = matToArray(matfile_HV, param_HV);
    img_mat[2] = matToArray(matfile_VV, param_VV);
    for(int i=0;i<3;i++){
        HostInitMat(img[i], h, w);
        cudaMemcpy2D(img[i].meta_data, img[i].pitch, img_mat[i], sizeof(cuComplex)*w, sizeof(cuComplex)*w, img[i].height, cudaMemcpyHostToDevice);
        imgPad[i] = HostPadMat(img[i], N, N);    // pad to use sliding windows
    }
    HostInitImg(resImg, h, w);
    dim3 griddim(16,16);
    dim3 blockdim(32,32);
    size_t size = (size_t)THREADSPACE*blockdim.x*blockdim.y*griddim.x*griddim.y;
    printf("%lu", size);
    cudaError_t error_t = cudaMalloc((void **)&MemPool, sizeof(char)*size); //分配的不是全局内存空间
    switch( error_t )
    {
      case cudaSuccess: printf("cudaSuccess\n");break;
      case cudaErrorMemoryAllocation:printf("cudaErrorMemoryAllocation\n");break;
      default:printf("default: %d \n",error_t );break;
    }
    for(int num = 0;num <= h/256;num++)
    {
        f_PWF<<<griddim,blockdim>>>(N, N, num, T);
        cudaDeviceSynchronize();
        printf("ok");
    }
    printf("%d\n", img[1].height);
    printf("%lu\n", sizeof(size_t));
    cudaDeviceSynchronize();
    Mat detect_res = Mat::zeros(h, w, CV_8UC1);
    cudaMemcpy2D(resImg_host, w*sizeof(char), resImg.meta_data, resImg.pitch, sizeof(char)*w, h, cudaMemcpyDeviceToHost);
    for(int i=0;i<h;i++)
    {
        for(int j=0;j<w;j++)
        {
            detect_res.at<uchar>(i,j) = resImg_host[INDEX(i, j, w)];
        }
    }
    imshow("detected" , detect_res);
    while(char(waitKey())!='q') 
	{    
    }
    detect_res.release();
    // for(int i=0;i<3;i++)
    // {
    //     HostDestroyMat(img[i]);
    // }
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