#pragma once
#include <iostream>
#include <functional>
#include <random>
#include <chrono>
#include <map>
#include <algorithm>
#include <stdexcept>  
#include <ctime>
#include <cstdlib>
#include <vector>
#include <memory>
#include <unordered_set>
#include <string>
#include <cctype>
#include <cuda_runtime.h>

using Matrix_d = std::vector<std::vector<float>>;
using vector_d = std::vector<float>;
using prob_id = std::vector<std::pair<float, int>>;

#define BLOCK_SIZE 32


__global__ void mmkernel(float *a, float *b, float *c, int m, int n, int p);
__global__ void tkernel(const float *a, float *b, int m, int n);


class Math
{
public:
  vector_d random_vector(const int & size, float scale = 1.0);
  float sqrt(const float & x, int approx = 10);
  int random_int(const int&a , const int& b);
  float max(const vector_d &vector);
  float min(const vector_d & vector);
  float ln(float n, int approx = 20);
  long long factorial_integer(int value);
  float power(float x, int n);
  float power_fractional(float x, float n);
  float exp(float x, int expansion_term=10);
  float combinatorics_choose(float x, int k);
  float sin(float y);
  float cos(float y);
  int TopKSampler(const vector_d& prob, const int &k);
  float ln_sub1(float y, int approx = 20);
  float round(float X, int decimal);
};

class Linear_Algebra
{


public:
  void printVector(const vector_d vector);
  void printMatrix(const Matrix_d & matrix);
  void printMatrixHead(const Matrix_d & matrix);
  float determinant(const Matrix_d & matrix, int row);
  void shape(Matrix_d A);
  Matrix_d rowEchelon(const Matrix_d& matrix);
  Matrix_d Diagonalize(const Matrix_d& matrix);
  Matrix_d remove_ij(const Matrix_d& matrix, int a, int b);
  Matrix_d MatMul(const Matrix_d& matrixA, const Matrix_d& matrixB);
  Matrix_d Inverse(const Matrix_d& matrix);
  Matrix_d Identity(const int& n);
  Matrix_d Transpose(const Matrix_d& matrix);
  Matrix_d add(const Matrix_d& matrixA, const Matrix_d&  matrixB);
  Matrix_d subtract(Matrix_d matrixA, Matrix_d matrixB);
  Matrix_d force_add(const Matrix_d& matrixA, const Matrix_d& matrixB);
  Matrix_d force_subtract(const Matrix_d& matrixA, const Matrix_d& matrixB);
  vector_d flatten(const Matrix_d& matrix);
  Matrix_d scalar_multiply(const Matrix_d& matrix, float scaler);
  float sum(const Matrix_d& matrix);
  Matrix_d multiply(Matrix_d matrixA, Matrix_d matrixB);
  Matrix_d force_multiply(const Matrix_d& matrixA, const Matrix_d& matrixB);
  Matrix_d accuracy(Matrix_d A, Matrix_d B);
  Matrix_d argmax(Matrix_d A, int axis=1);
  Matrix_d argmin(Matrix_d A, int axis=1);
  bool isEigenValue(Matrix_d matrix, float lambda);
  bool isEigenVector(Matrix_d matrix, vector_d vect);
  vector_d eigenVector(const Matrix_d & matrix);
  vector_d vectorsum( const vector_d & v1, const vector_d& v2);
  Matrix_d RandMatrix(const int &row, const int &col, int scale);
  Matrix_d RandMatrixXavier(const int &row, const int &col);
  float mean(const vector_d &z);
  float std(const vector_d &z, const float& error);
  Matrix_d LayerNorm(Matrix_d Z, float error = 1e-4f);
};

class Linear_Algebra_GPU
{
private:
    // Assuming you have multiple CUDA streams for parallel execution
    std::vector<cudaStream_t> streams;
    int num_heads;
    void CudaT(const float *a, float *b, int m, int n);
    void CudaMM(float *a, float *b, float *c, int m, int n, int p, int stream_id);
public:
    void printVector(const vector_d vector);
    void printMatrix(const Matrix_d& matrix);
    void printMatrixHead(const Matrix_d& matrix);
    Matrix_d Transpose(const Matrix_d & matrixA);
    cudaStream_t get_stream(int stream_id);
    Matrix_d MatMul(const Matrix_d & matrixA, const Matrix_d & matrixB, int stream_id = 0);
    void InitializeStreams(int num_streams = 1);
    Linear_Algebra_GPU();
    ~Linear_Algebra_GPU();
    size_t stream_count();
};

class Machine_Learning
{
public:
  float max(vector_d vector);
  float min(vector_d vector);
  long long factorial_integer(int value);
  float exp(float x, int expansion_term=10);
  std::function<float(float)>  derivative(std::function<float(float)> function, int order = 1, float h = 1e-5);
  float power(float x, int n);
  std::function<float(float)> loss(std::function<float(float)> function);
  vector_d single_training( std::function<float(float)> function, int epochs = 1000, float lr = 1e-2);
  Matrix_d RelU(Matrix_d matrix);
  Matrix_d deriv_ReLU(Matrix_d matrix);
  vector_d softmax(vector_d Z);
  Matrix_d SoftMax(Matrix_d Z);
  Matrix_d deriv_SoftMax(Matrix_d Z);
  vector_d deriv_softmax(vector_d Z);
  vector_d softmask(vector_d Z, int t);
  vector_d deriv_softmask(vector_d Z, int t);
  Matrix_d SoftMask(Matrix_d Z);
  Matrix_d deriv_SoftMask(Matrix_d Z);

};

