# 任务计划
+ 下一步任务，代码重构，将cuMat封装为c++类, 细分为cuHostMat与cuDevMat
+ cuHostMat 采用二维数组，而cuDevMat使用线程的空间
+ 可以通过该种形式访问元素 mat.index(i,j) 注意通过mallocpitch分配的每行空间