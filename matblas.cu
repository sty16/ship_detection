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

__global__ void cuMatHer(cuMat a, cuMat res, cuComplex alpha){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if(i < res.height && j < res.width){
        res.meta_data[INDEX(i, j, res.width)] = make_cuComplex(0, 0);
        for(int k = 0;k < a.width; k++){
            res.meta_data[INDEX(i, j, res.width)] = cuCaddf(res.meta_data[INDEX(i, j, res.width)], cuCmulf(a.meta_data[INDEX(i, k, a.width)], cuConjf(a.meta_data[INDEX(j, k, a.width)])));    
        }
        res.meta_data[INDEX(i, j, res.width)] = cuCmulf(alpha, res.meta_data[INDEX(i, j, res.width)]);
    }
}

__global__ void  cuMatPad(cuMat a, cuMat res, int pad_row, int pad_col){
    int i = threadIdx.x + blockDim.x * blockIdx.x;  // the ith row
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if(i<res.height && j < res.width){
        if(i<pad_row){
            if(j<pad_col){
                res.data[i][j] = a.data[pad_row - 1 - i][pad_col - 1 -j];
            }
            else if(j >= pad_col && j<pad_col+a.width){
                res.data[i][j] = a.data[pad_row - 1 -i][j-pad_col];
            }
            else{
                res.data[i][j] = a.data[pad_row - 1 - i][2*a.width+pad_col-1-j];
            }
        }else if(i < pad_row + a.height && i >= pad_row){
            if(j<pad_col){
                res.data[i][j] = a.data[i-pad_row][pad_col - 1 -j];
            }
            else if(j >= pad_col && j<pad_col+a.width){
                res.data[i][j] = a.data[i-pad_row][j-pad_col];
            }
            else{
                res.data[i][j] = a.data[i-pad_row][2*a.width+pad_col-1-j];
            }
        }else{
            if(j<pad_col){
                res.data[i][j] = a.data[2*a.height+pad_row-1-i][pad_col - 1 -j];
            }
            else if(j >= pad_col && j<pad_col+a.width){
                res.data[i][j] = a.data[2*a.height+pad_row-1-i][j-pad_col];
            }
            else{
                res.data[i][j] = a.data[2*a.height+pad_row-1-i][2*a.width+pad_col-1-j];
            }
       }
    }
}

__global__ void transposeSmem(cuMat a, cuMat res){                  // use shared memory to transpose matrix
    int i = threadIdx.x + blockDim.x * blockIdx.x;  
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    __shared__ cuComplex tile[BDIMX][BDIMY];
    int row, col, trow, tcol;
    int m = a.height/blockDim.x;        // the number of full filled block
    int n = a.width/blockDim.y;
    if(blockIdx.x < m && blockIdx.y < n)
    {                                                                              // full block and non-full block
        tile[threadIdx.x][threadIdx.y] = a.data[i][j]; 
        int numx;                       // find the index  
        numx = threadIdx.x*blockDim.y + threadIdx.y;
        trow = numx / blockDim.x;
        tcol = numx % blockDim.x;
        row = trow + blockIdx.y*blockDim.y;
        col = tcol + blockIdx.x*blockDim.x;
    }else{
        row = j;col = i;
    }
   __syncthreads();                                                                  //wait for the tile filled with value;
    if(row<res.height && col<res.width){
        if(blockIdx.x < m && blockIdx.y < n){
            res.data[row][col] = cuConjf(tile[tcol][trow]);                                        //coalesced  write
        }else{
            res.data[row][col] = cuConjf(a.data[i][j]);
        }
    }
}

__global__ void transposeDmem(cuMat a, cuMat res){
    int i = threadIdx.x + blockDim.x * blockIdx.x;  
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    __shared__ cuComplex tile[BDIMX][BDIMY];
    int row, col, trow, tcol;
    int m = a.height/blockDim.x;        // the number of full filled block
    int n = a.width/blockDim.y;
    if(blockIdx.x < m && blockIdx.y < n)
    {                                                                              // full block and non-full block
        tile[threadIdx.x][threadIdx.y] = a.meta_data[INDEX(i, j, a.width)]; 
        int numx;                       // find the index  
        numx = threadIdx.x*blockDim.y + threadIdx.y;
        trow = numx / blockDim.x;
        tcol = numx % blockDim.x;
        row = trow + blockIdx.y*blockDim.y;
        col = tcol + blockIdx.x*blockDim.x;
    }else{
        row = j;col = i;
    }
   __syncthreads();                                                                  //wait for the tile filled with value;
    if(row<res.height && col<res.width){
        if(blockIdx.x < m && blockIdx.y < n){
            res.meta_data[INDEX(row, col, res.width)] = cuConjf(tile[tcol][trow]);                                        //coalesced  write
        }else{
            res.meta_data[INDEX(row, col, res.width)] = cuConjf(a.meta_data[INDEX(i, j, a.width)]);
        }
    }
}

__device__ void InitMat(cuMat &mat, int h, int w){
    mat.height = h;
    mat.width = w;
    cudaMalloc((void**)&mat.data, sizeof(cuComplex *)*h);
    cudaMalloc((void**)&mat.meta_data, sizeof(cuComplex)*h*w);
    for(int i=0;i<h;i++){
        mat.data[i] = mat.meta_data + i*w;
    }
}

__host__ void HostInitMat(cuMat &mat, int h, int w){
    mat.height = h;
    mat.width = w;
    cudaMallocManaged((void**)&mat.data, sizeof(cuComplex *)*h);
    cudaMallocPitch((void**)&mat.meta_data, &mat.pitch ,sizeof(cuComplex)*w, h);   //采用cudaMallocPitch分配2D数组加快访问
    for(size_t i=0;i<h;i++)
    {
        mat.data[i] =  (cuComplex *)((char *)mat.meta_data + i*mat.pitch);     //直接访问设备内存会报错，使用cudaMallocManaged
    }
    // Memory does not need to be initialized to ensure speed
}

__device__ void DeviceInitMat(cuMat &mat, char *begin, int &pointer, int h, int w){
    // begin 线程数据起使地址 pointer当前的指针字节位置
    mat.meta_data = (cuComplex *)((char *)begin + pointer); //分配矩阵地址
    pointer = pointer + h*w*sizeof(cuComplex);    // 指针进行偏移
    if(pointer >=  THREADSPACE){
        printf("ErrorMallocAllocation\n");
        mat.height = 0;
        mat.width = 0;
        mat.meta_data = (cuComplex *)begin;      //回到起始空间
    }else{
        mat.height = h;
        mat.width = w;
    }
}

__host__ void HostInitImg(cuImg &img, int h, int w){
    img.height = h;
    img.width = w;
    cudaMallocManaged((void **)&img.data, sizeof(uint8 *)*h);
    cudaMallocPitch((void **)&img.meta_data, &img.pitch, sizeof(uint8)*w, h);
    for(size_t i = 0;i<h;i++)
    {
        img.data[i] = (uint8 *)((char *)img.meta_data + i*img.pitch);
    }
}

__device__ void DestroyMat(cuMat &mat){
    cudaFree(mat.data);
    cudaFree(mat.meta_data);
}

__host__ void HostDestroyMat(cuMat &mat){
    cudaFree(mat.data);
    cudaFree(mat.meta_data);
}

__host__ void HostDestroyImg(cuImg &img){
    cudaFree(img.data);
    cudaFree(img.meta_data);
}

__device__ void DeviceDestroyMat(cuMat mat, char *begin, int &pointer)
{
    pointer = pointer - mat.height*mat.width*sizeof(cuComplex);
    cuComplex *temp = (cuComplex *)((char *)begin + pointer);
    if(temp != mat.meta_data)
    {
        printf("cudaFreeFailure\n");
        pointer = pointer + mat.height*mat.width*sizeof(cuComplex);   //线程空间相当于栈空间，注意先进后出的释放顺序
    }
}

__device__  cuMat  MulMat(cuMat a, cuMat b, cuComplex alpha){
   cuMat res; 
   if(a.width == b.height){
        InitMat(res, a.height, b.width);
        dim3 blockdim(16, 16);
        dim3 griddim(a.height/16 + 1, b.width/16 + 1);
        cuMatMul<<<griddim,blockdim>>>(a, b, res, alpha);    // Compute matrix multiplication in parallel
        cudaDeviceSynchronize();   // parent kernel waits for child kernel 
        return res;
   }else{
        printf("the size of two input Matrix are not match\n");
        InitMat(res, 1, 1);
        return res;
   }
}

__host__ cuMat HostPadMat(cuMat a, int pad_row, int pad_col){
    cuMat res;
    HostInitMat(res, a.height + 2*pad_row, a.width + 2*pad_col);   // 主机调用, 分配设备内存
    dim3 blockdim(32, 32);
    dim3 griddim((int)(res.height/32 + 1), (int)(res.width/32 + 1));                         // pad by row
    cuMatPad<<<griddim,blockdim>>>(a, res, pad_row, pad_col);
    cudaDeviceSynchronize(); 
    return res;
}

__device__ cuMat TransposeMat(cuMat a){
    cuMat res;
    InitMat(res, a.width, a.height);
    dim3 blockdim(32, 32);
    dim3 griddim(a.height/32 + 1, a.width/32 + 1);
    transposeSmem<<<griddim, blockdim>>>(a, res); 
    cudaDeviceSynchronize();
    printf("%d", res.height);
    return res;
} 

__device__ cuMat DeviceTransMat(cuMat a, char *begin, int &pointer){
    cuMat res;
    DeviceInitMat(res, begin, pointer, a.width, a.height);
    dim3 blockdim(32, 32);
    dim3 griddim(a.height/32 + 1, a.width/32 + 1);
    transposeSmem<<<griddim, blockdim>>>(a, res); 
    cudaDeviceSynchronize();
    printf("%d", res.height);
    return res;
} 

__device__ cuMat HerMat(cuMat mat, char *begin, int &pointer, cuComplex alpha)
{
    cuMat res;
    DeviceInitMat(res, begin, pointer, mat.height, mat.height);
    dim3 blockdim(16, 16);
    dim3 griddim(res.height/16 + 1, res.width/16 + 1);
    cuMatHer<<<griddim, blockdim>>>(mat, res, alpha);
    cudaDeviceSynchronize();
    return res;
}

__device__ cuComplex MatDet(cuMat mat, char *begin, int &pointer)
{
    if(mat.height != mat.width)
    {
        printf("the height and width of the matrix are not match\n");
        return make_cuComplex(0, 0);
    }
    if(mat.height == 1)
    {
        cuComplex det = mat.meta_data[INDEX(0, 0, mat.width)];
        return det;
    }
    cuMat temp;
    DeviceInitMat(temp, begin, pointer, mat.height-1, mat.width-1);
    cuComplex det = make_cuComplex(0, 0);
    int row, col;
    for(int i = 0; i < mat.height; i++)
    {
        for(int j = 0; j < mat.height - 1; j++)
        {
            for(int k = 0; k < mat.width - 1; k++)
            {
                row = j + 1;
                col = (k>=i)?k+1:k;
                temp.meta_data[INDEX(j, k, temp.width)] = mat.meta_data[INDEX(row, col, mat.width)];
            }
        }
        cuComplex cofactor = MatDet(temp, begin, pointer);
        // printf("%f,%f\n", cofactor.x, cofactor.y);
        if(i%2 == 0)
        {
            det = cuCaddf(det, cuCmulf(mat.meta_data[INDEX(0, i, mat.width)], cofactor));
        }else{
            det = cuCsubf(det, cuCmulf(mat.meta_data[INDEX(0, i, mat.width)], cofactor));
        }
    }
    DeviceDestroyMat(temp, begin, pointer);    // 释放递归过程中分配的线程栈空间
    return det;
}