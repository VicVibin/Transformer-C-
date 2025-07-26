#include "includes/MathGPU.h"



__global__ void mmkernel(float *a, float *b, float *c, int m, int n, int p)
{
int row = threadIdx.y + blockIdx.y * blockDim.y;
int col = threadIdx.x + blockIdx.x * blockDim.x;

// Shared memory tiles
__shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
__shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

float sum = 0.0f;

for (int t = 0; t < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; t++)
{
    if (row < m && (t * BLOCK_SIZE + threadIdx.x) < n)
        s_A[threadIdx.y][threadIdx.x] = a[row * n + t * BLOCK_SIZE + threadIdx.x];
    else
        s_A[threadIdx.y][threadIdx.x] = 0.0f;

    if (col < p && (t * BLOCK_SIZE + threadIdx.y) < n)
        s_B[threadIdx.y][threadIdx.x] = b[(t * BLOCK_SIZE + threadIdx.y) * p + col];
    else
        s_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; ++k)
    {
        sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
    }


    __syncthreads();
}

if (row < m && col < p)
{
    c[row * p + col] = sum;   
}

}

__global__ void tkernel(const float *a, float *b, int m, int n)
{
 __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1]; // Avoid bank conflicts
 int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
 int y  =  blockIdx.y * BLOCK_SIZE + threadIdx.y;

 if (x < n && y < m)
 {

    tile[threadIdx.y][threadIdx.x] = a[y * n + x];
 }

 __syncthreads();

 x = blockIdx.y * BLOCK_SIZE + threadIdx.x; // transpose block offset
 y = blockIdx.x * BLOCK_SIZE + threadIdx.y;


  if (x < m && y < n)
 {

    b[y * m + x] = tile[threadIdx.x][threadIdx.y];
 }
}

void Linear_Algebra::printVector(const vector_d vector)
    {
        for (float val: vector)
        {
            std::cout<< val << " "; 
        }
        std::cout << " " << std::endl;
        std::cout << "---------------------------\n";
    }

    // Function to print the matrix 
void Linear_Algebra::printMatrix(const Matrix_d& matrix) 
    {
        for (const auto& row : matrix) 
        {
            for (float val : row) 
            {
                std::cout << val << "\t";  // Print each value with a tab space
            }
        std::cout << std::endl;  // Newline after each row
        }
        std::cout << "-----------------------------------\n";  // Separator for clarity

    }

void Linear_Algebra::printMatrixHead(const Matrix_d& matrix) 
    {   int row = matrix.size();
        int col = matrix[0].size();

        for (int i =0; i < std::min(5, row); i++)
        {
            for (int j = 0; j < std::min(5, col); j++)
            {
                std::cout << matrix[i][j] << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << " ----------------------------------------\n"; // Seperate matrix for clarity
        

    }

float Linear_Algebra::determinant(const Matrix_d& matrix, int row_swaps = 0) 
    {
        int rows = matrix.size();
        int cols = matrix[0].size();

        if(rows != cols)
        {
            std::cout << "This is not a square matrix" <<std::endl;
            return 1;
        }
        
        // Diagonalizing Phase
        Matrix_d mat = matrix;
        int row_s = row_swaps;
        for (int i = 0; i < rows - 1; i++) 
        {
            for (int j = i + 1; j < rows; j++) {  // Ensure we're within row bounds
                if (mat[i][i] == 0) 
                { // Avoid division by zero
                    std::cerr << "Error: Division by zero encountered.\n";
                    vector_d w = mat[i];
                    mat[i] = mat[i+1];
                    mat[i+1] = w;
                    printMatrixHead(mat);
                    row_s++;
                    return determinant(mat, row_s);
                }
                float multiplier = (float)mat[j][i] / mat[i][i];
                for (int k = 0; k < cols; k++){
                    float n = mat[j][k];
                    float n1 = mat[i][k];
                    mat[j][k] = multiplier * n1 - n;

                }
            }
        }

        float d = 1;
        for (int i = 0; i < rows; i++)
        {
            int t = -1;
            if (i % 2 == 1)
            {
                t = 1;
            }
            d *= t * mat[i][i];
        }
        if (row_s % 2 == 0)
        {
            return d;
        }
        else
        {
            return -d;
        }
    }
    
// Dummy function to initialize random weights
Matrix_d Linear_Algebra::RandMatrix(const int & rows, const int & cols, int scale) 
{
    Math m;
    Matrix_d weights(rows, vector_d(cols));
    float value = m.sqrt(6 / (rows + cols));
    for (int i = 0; i < rows; ++i)
        weights[i] = m.random_vector(cols, scale);
    return weights;
}

Matrix_d Linear_Algebra::RandMatrixXavier(const int & rows, const int & cols) 
{
    Math m;
    Matrix_d weights(rows, vector_d(cols));
    float value = 6 / m.sqrt(rows + cols);
    for (int i = 0; i < rows; ++i)
        weights[i] = m.random_vector(cols, value);
    return weights;
}  

Matrix_d Linear_Algebra::rowEchelon(const Matrix_d& matrix) 
    {
        int rows = matrix.size();
        int cols = matrix[0].size();

        // Forward Phase
        Matrix_d mat = matrix;

        for (int i = 0; i < rows - 1; i++) {
            for (int j = i + 1; j < rows; j++) {  // Ensure we're within row bounds
                if (mat[i][i] == 0) { // Avoid division by zero
                    std::cerr << "Error: Division by zero encountered.\n";
                    printMatrixHead(mat);
                    return mat;
                }
                float multiplier = (float)mat[j][i] / mat[i][i];
                for (int k = 0; k < cols; k++){
                    float n = mat[j][k];
                    float n1 = mat[i][k];
                    mat[j][k] = multiplier * n1 - n;

                }
            }
        }
        std::cout<<"Diagonalized_Matrix"<<std::endl;
        printMatrixHead(mat);
        int t = rows;

        if (rows > cols)
        {
            t = cols;
        }

        // Backward Phase
        for (int i = t - 1; i > 0; i--) {
            for (int j = i - 1; j > -1; j--) {  // Ensure we're within row bounds
                if (mat[i][i] == 0) 
                { // Avoid division by zero
                    printMatrixHead(mat);
                    std::cerr << "Error: Division by zero encountered.\n";
                    return mat;
                }
                float multiplier = (float)mat[j][i] / mat[i][i];
                for (int k = cols - 1; k > -1; k--)
                {
                    float n = mat[j][k];
                    float n1 = mat[i][k];
                    mat[j][k] = multiplier * n1 - n;

                }
            }
        }
        for (int i = 0; i < rows; i++){
            float scaler = mat[i][i];
            for (int j = 0; j < cols; j++){
                mat[i][j] = mat[i][j] / scaler;
            }
        }
        std::cout<<"Finished Echelon"<<std::endl;
        printMatrixHead(mat);

        return mat;
    }

Matrix_d Linear_Algebra::Diagonalize(const Matrix_d& matrix) 
    {
        int rows = matrix.size();
        int cols = matrix[0].size();

        // Forward Phase
        Matrix_d mat = matrix;

        for (int i = 0; i < rows - 1; i++) {
            for (int j = i + 1; j < rows; j++) {  // Ensure we're within row bounds
                if (mat[i][i] == 0) { // Avoid division by zero
                    std::cerr << "Error: Division by zero encountered.\n";
                    printMatrixHead(mat);
                    return mat;
                }
                float multiplier = (float)mat[j][i] / mat[i][i];
                for (int k = 0; k < cols; k++){
                    float n = mat[j][k];
                    float n1 = mat[i][k];
                    mat[j][k] = multiplier * n1 - n;

                }
            }
        }
        std::cout<<"Diagonalized_Matrix"<<std::endl;
        printMatrixHead(mat);
        return mat;
    }

Matrix_d Linear_Algebra::remove_ij(const Matrix_d& matrix, int a, int b)
    {
        int row = matrix.size();
        int col = matrix[0].size();
        Matrix_d data(row-1, vector_d(col-1, 0));
        int newr = 0;
        for(int i = 0; i < row; i++)
        {   
            int newc = 0;
            if(i == a)
            {
                continue;
            }

            for(int j = 0; j < col; j++)
            {   

                if(j == b)
                {
                    continue;
                }
                data[newr][newc] = matrix[i][j];
                newc++;

            }
            newr++;
        }
        return data;

    }

Matrix_d Linear_Algebra::MatMul(const Matrix_d& matrixA, const Matrix_d& matrixB) // Can be made parallel
    {   
        int rA = matrixA.size();
        int CA = matrixA[0].size();
        int rB = matrixB.size();
        int CB = matrixB[0].size();
        if (CA != rB){
            std:: cerr << "Error: Matrix Sizes do not match.\n";
            return {{}};
        }

        Matrix_d matrix(rA, vector_d(CB, 0.0f));;

        for (int i = 0; i < CB; i++)
        {

            for (int j = 0; j < rA; j++)
            {
                float cumprod = 0;

                for (int k = 0; k < rB; k++)
                {  
                    cumprod += matrixA[j][k] * matrixB[k][i];
                }
                matrix[j][i] = cumprod;
                
            }
        }
    return matrix;

    }

Matrix_d Linear_Algebra::Inverse(const Matrix_d& matrix) // Can be made parallel
    {
        int row = matrix.size();
        int col = matrix[0].size();
        if (row !=col)
        {
            std::cout << "The matrix is not a square matrix, cannot find the inverse" << std::endl;
            return {{}};
        }
        
        Matrix_d inverse(row, vector_d(col));
        Linear_Algebra lin;
        float determinant_factor = 1 / lin.determinant(matrix);
        std::cout << determinant_factor << std::endl;

        for(int i = 0; i < row; i++) // Using the Adj(A) algorithm
        {
            for(int j = 0; j < col; j++)
            {   
                if((i + j) % 2 == 0)
                {
                    inverse[j][i] =  -1 * determinant_factor * lin.determinant(lin.remove_ij(matrix, i ,j));
                }
                else
                {
                    inverse[j][i] = determinant_factor * lin.determinant(lin.remove_ij(matrix,i,j));
                }
            }
        }
        return inverse;
    }

Matrix_d Linear_Algebra::Identity(const int& n)
    {
        Matrix_d id(n, vector_d(n));
        for(int i = 0; i < n; i++)
        {
            for(int j  = 0; j < n; j++)
            {
                if( i == j)
                {
                    id[i][j] = 1;
                }
                else
                {
                    id[i][j] = 0;
                }
            }
        }
        return id;
    }

Matrix_d Linear_Algebra::Transpose(const Matrix_d& matrix)
    {
        int row = matrix.size();
        int Col = matrix[0].size();

        Matrix_d Matrix(Col, vector_d(row, 0.0f));

        for(int i = 0; i < Col; i++)
        {
            for(int j = 0; j < row; j++)
            {
                Matrix[i][j] = matrix[j][i];
            }
        }
        return Matrix;
    }
    
Matrix_d Linear_Algebra::add(const Matrix_d& matrixA, const Matrix_d&  matrixB)
    {
        int t = matrixA[0].size();
        int t1 = matrixB[0].size();
        int a = matrixA.size();
        int a1 = matrixB.size();


        Matrix_d Addition(a, vector_d(t));
        if( t == t1 && a == a1)
        {
            for(int i = 0; i < a; i++)
            {
                for(int j = 0; j < t; j++)
                {
                    Addition[i][j] = matrixA[i][j] + matrixB[i][j];
                }
            }
            return Addition;
        }
        else
        {
            std::cout << "SIZE MISMATCH \n" << std::endl;
            std::cout << "Cannot add matrices of sizes[" << a << ", " << t << "]" << " and [" 
            << a1 << ", " << t1 << "]" << std::endl;
            return {{}};
        }

        
    }

Matrix_d Linear_Algebra::subtract(Matrix_d matrixA, Matrix_d matrixB)
    {
        int t = matrixA[0].size();
        int t1 = matrixB[0].size();
        int a = matrixA.size();
        int a1 = matrixB.size();

        if( t == t1 && a == a1)
        {
            for(int i = 0; i < a; i++)
            {
                for(int j = 0; j < t; j++)
                {
                    matrixA[i][j] -= matrixB[i][j];
                }
            }
            return matrixA;
        }
        else
        {
            std::cout << "SIZE MISMATCH \n" << std::endl;
            std::cout << "Cannot subtract matrices of sizes[" << a << ", " << t << "]" << " and [" 
            << a1 << ", " << t1 << "]" << std::endl;
            return {{}};
        }
    }

Matrix_d Linear_Algebra::force_add(const Matrix_d& matrixA, const Matrix_d& matrixB)
    {
        int t = matrixA[0].size();
        int t1 = matrixB[0].size();
        int a = matrixA.size();
        int a1 = matrixB.size();

        Matrix_d Addition(a, vector_d(t));

        if( t == t1 && a == a1)
        {
            for(int i = 0; i < a; i++)
            {
                for(int j = 0; j < t; j++)
                {
                    Addition[i][j] = matrixA[i][j] + matrixB[i][j];
                }
            }
            return Addition;
        }
         // if a = m x n and b  = m x 1
        else if(a == a1 && t1 == 1)
        {
            for(int i = 0; i < a; i++)
            {
                for(int j = 0; j < t; j++)
                {
                    Addition[i][j] = matrixA[i][j] + matrixB[i][0];
                }
            }
            return Addition;  
        }
        // if a = m x n and b  = 1 x 1
        else if(a1 == 1 && t1 == 1)
        {
            for(int i = 0; i < a; i++)
            {
                for(int j = 0; j < t; j++)
                {
                    Addition[i][j] = matrixA[i][j] + matrixB[0][0];
                }
            }
    
            return Addition;  
        }

        // if a = m x n and b  = 1 x n
        else if(a1 == 1 && t1 == t)
        {
            for(int i = 0; i < a; i++)
            {
                for(int j = 0; j < t; j++)
                {
                    Addition[i][j] = matrixA[i][j] + matrixB[0][j];
                }
            }

            return Addition;  
        }

        else
        {
            std::cout << "SIZE MISMATCH \n" << std::endl;
            std::cout << "Cannot force_add matrices of sizes [" << a << ", " << t << "]" << " and [" 
            << a1 << ", " << t1 << "]" << std::endl;
            return {{}};
        }
    
    }

Matrix_d Linear_Algebra::force_subtract(const Matrix_d& matrixA, const Matrix_d& matrixB)
    {
        int t = matrixA[0].size();
        int t1 = matrixB[0].size();
        int a = matrixA.size();
        int a1 = matrixB.size();

        Matrix_d m(a, vector_d(t));
        if( t == t1 && a == a1)
        {
            for(int i = 0; i < a; i++)
            {
                for(int j = 0; j < t; j++)
                {
                    m[i][j] = matrixA[i][j] - matrixB[i][j];
                }
            }
            return m;
        }
        else if(a == a1 && t1 == 1)
        {
            for(int i = 0; i < a; i++)
            {
                for(int j = 0; j < t; j++)
                {
                    m[i][j]= matrixA[i][j] - matrixB[i][0];
                }
            }
            return m;  
        }

        else if(a1 == 1 && t1 == 1)
        {
            for(int i = 0; i < a; i++)
            {
                for(int j = 0; j < t; j++)
                {
                    m[i][j]= matrixA[i][j] - matrixB[0][0];
                }
            }
            return m;  
        }

        else if(a1 == 1 && t1 == t)
        {
            for(int i = 0; i < a; i++)
            {
                for(int j = 0; j < t; j++)
                {
                    m[i][j] = matrixA[i][j] - matrixB[0][j];
                }
            }
            return m;  
        }


        else
        {
            std::cout << "SIZE MISMATCH \n" << std::endl;
            std::cout << "Cannot force subtract matrices of sizes [" << a << ", " << t << "]" << " and [" 
            << a1 << ", " << t1 << "]" << std::endl;
            return {{}};
        }
    }

vector_d Linear_Algebra::flatten(const Matrix_d& matrix)
    {
        int row = matrix.size();
        int col = matrix[0].size();
        int sum = 0;

        vector_d var(row * col);;
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                var[sum] = matrix[i][j];
                sum += 1;
            }
        }
        return var;
    }

Matrix_d Linear_Algebra::scalar_multiply(const Matrix_d& matrix, float scaler)
    {
        int row  = matrix.size();
        int col  = matrix[0].size();

        Matrix_d v(row, vector_d(col));

        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                v[i][j] = matrix[i][j] * scaler;
            }
        }
        return v;
    }

float Linear_Algebra::sum(const Matrix_d& matrix)
    {   
        float num = 0;
        int row = matrix.size();
        int col = matrix[0].size();
        for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
            {
                num += matrix[i][j];
            }
        }
        return num;
    }

Matrix_d Linear_Algebra::multiply(Matrix_d matrixA, Matrix_d matrixB)
    {
        int t = matrixA[0].size();
        int t1 = matrixB[0].size();
        int a = matrixA.size();
        int a1 = matrixB.size();

        Matrix_d m(a, vector_d(t));

        if( t == t1 && a == a1)
        {
            for(int i = 0; i < a; i++)
            {
                for(int j = 0; j < t; j++)
                {
                    m[i][j] = matrixA[i][j] * matrixB[i][j];
                }
            }
            return m;
        }

        else
        {
            std::cout << "SIZE MISMATCH \n" << std::endl;
            std::cout << "Cannot subtract matrices of sizes[" << a << ", " << t << "]" << " and [" 
            << a1 << ", " << t1 << "]" << std::endl;
            return {{}};
        }
    }

Matrix_d Linear_Algebra::force_multiply(const Matrix_d& matrixA, const Matrix_d& matrixB)
    {
        int t = matrixA[0].size();
        int t1 = matrixB[0].size();
        int a = matrixA.size();
        int a1 = matrixB.size();

        Matrix_d mult(a, vector_d(t));

        if( t == t1 && a == a1)
        {
            for(int i = 0; i < a; i++)
            {
                for(int j = 0; j < t; j++)
                {
                    mult[i][j] = matrixA[i][j] * matrixB[i][j];
                }
            }
            return mult;
        }
        else if(a == a1 && t1 == 1)
        {
            for(int i = 0; i < a; i++)
            {
                for(int j = 0; j < t; j++)
                {
                    mult[i][j] = matrixA[i][j] * matrixB[i][0];
                }
            }
            return mult;  
        }

        else if(a1 == 1 && t1 == 1)
        {
            for(int i = 0; i < a; i++)
            {
                for(int j = 0; j < t; j++)
                {
                    mult[i][j] =  matrixA[i][j] * matrixB[0][0];
                }
            }
            return mult;  
        }

        else if(a1 == 1 && t1 == t)
        {
            for(int i = 0; i < a; i++)
            {
                for(int j = 0; j < t; j++)
                {
                    mult[i][j] = matrixA[i][j] * matrixB[0][j];
                }
            }
            return mult;  
        }


        else
        {
            std::cout << "SIZE MISMATCH \n" << std::endl;
            std::cout << "Cannot force subtract matrices of sizes [" << a << ", " << t << "]" << " and [" 
            << a1 << ", " << t1 << "]" << std::endl;
            return {{}};
        }
    }
    
Matrix_d Linear_Algebra::accuracy(Matrix_d A, Matrix_d B)
    {
        int rA = A.size(), cA = A[0].size(), rB = B.size(), cB = B[0].size();
        Matrix_d C(rA, vector_d(cA, 0));
        if (rA == rB && cA == cB)
        {
            for (int i = 0; i < rA; i++)
            {
                for (int j = 0; j < cA; j++)
                {
                    if(A[i][j] == B[i][j])
                    {
                        C[i][j] = 1;
                    }
                }
            }
        return C;
        }
        else 
        {
            std::cout << " The matrices are not the same size, test for equality cannot hold" << std::endl;
        return {{0}};
        }
    }

void Linear_Algebra::shape(Matrix_d A)
{
    int m  = A.size();
    int n = A[0].size();
    std::cout << "Shape of matrix: " << m << " x " << n << "\n";
}

Matrix_d Linear_Algebra::argmax(Matrix_d A, int axis)
    {
        int row = A.size(), col = A[0].size();

        if (axis == 1)
        {
            // Return the index of the maximum value for each column
            Matrix_d vector(1, vector_d(col));
            for (int i = 0; i < col; i++)
            {
                float max_val = A[0][i];
                int max_idx = 0;
                for (int j = 1; j < row; j++)
                {
                    if (A[j][i] > max_val)
                    {
                        max_val = A[j][i];
                        max_idx = j;
                }
            }
            vector[0][i] = max_idx; // Store the index, not the value
        }
        return vector;
    }
    else if (axis == 0)
    {
        // Return the index of the maximum value for each row
        Matrix_d vector(1, vector_d(row));
        for (int i = 0; i < row; i++)
        {
            float max_val = A[i][0];
            int max_idx = 0;
            for (int j = 1; j < col; j++)
            {
                if (A[i][j] > max_val)
                {
                    max_val = A[i][j];
                    max_idx = j;
                }
            }
            vector[0][i] = max_idx; // Store the index, not the value
        }
        return vector;
    }
    else
    {
        std::cout << "Assuming Axis = 1, use proper axis input" << std::endl;
        // Default to axis=1 behavior
        Matrix_d vector(1, vector_d(col));
        for (int i = 0; i < col; i++)
        {
            float max_val = A[0][i];
            int max_idx = 0;
            for (int j = 1; j < row; j++)
            {
                if (A[j][i] > max_val)
                {
                    max_val = A[j][i];
                    max_idx = j;
                }
            }
            vector[0][i] = max_idx; // Store the index, not the value
        }
        return vector;
    }
    }

Matrix_d Linear_Algebra::argmin(Matrix_d A, int axis)
    {
        int row = A.size(), col = A[0].size();

        if (axis == 1)
        {
            // Return the index of the maximum value for each column
            Matrix_d vector(1, vector_d(col));
            for (int i = 0; i < col; i++)
            {
                float max_val = A[0][i];
                int max_idx = 0;
                for (int j = 1; j < row; j++)
                {
                    if (A[j][i] < max_val)
                    {
                        max_val = A[j][i];
                        max_idx = j;
                }
            }
            vector[0][i] = max_idx; // Store the index, not the value
        }
        return vector;
    }
    else if (axis == 0)
    {
        // Return the index of the maximum value for each row
        Matrix_d vector(1, vector_d(row));
        for (int i = 0; i < row; i++)
        {
            float max_val = A[i][0];
            int max_idx = 0;
            for (int j = 1; j < col; j++)
            {
                if (A[i][j] < max_val)
                {
                    max_val = A[i][j];
                    max_idx = j;
                }
            }
            vector[0][i] = max_idx; // Store the index, not the value
        }
        return vector;
    }
    else
    {
        std::cout << "Assuming Axis = 1, use proper axis input" << std::endl;
        // Default to axis=1 behavior
        Matrix_d vector(1, vector_d(col));
        for (int i = 0; i < col; i++)
        {
            float max_val = A[0][i];
            int max_idx = 0;
            for (int j = 1; j < row; j++)
            {
                if (A[j][i] < max_val)
                {
                    max_val = A[j][i];
                    max_idx = j;
                }
            }
            vector[0][i] = max_idx; // Store the index, not the value
        }
        return vector;
    }
    }

bool Linear_Algebra::isEigenValue(Matrix_d matrix, float lambda)
    {   
        int row = matrix.size();
        int col = matrix[0].size();

        if (row != col)
        {
            std::cout << "Requires N x N matrix for operation to be viable" << std::endl;
            return false;
        }

        Linear_Algebra lin;
        Matrix_d characteristic = lin.force_subtract(matrix, lin.scalar_multiply(lin.Identity(row), lambda));
        float determinant = lin.determinant(characteristic);
        if (determinant == 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

bool Linear_Algebra::isEigenVector(Matrix_d matrix, vector_d vect)
    {   
        int row = matrix.size();
        int col = matrix[0].size();

        if (row != col)
        {
            std::cout << "Requires N x N matrix for operation to be viable" << std::endl;
            return false;
        }
        Linear_Algebra lin;
        for (int i = 0; i < vect.size(); i++)
        {   
            
            Matrix_d characteristic = lin.force_subtract(matrix, lin.scalar_multiply(lin.Identity(row), vect[i]));
            float determinant = lin.determinant(characteristic);
            if (determinant != 0)
            {
                return false;
            }
        }
        return true;
    }

vector_d Linear_Algebra::eigenVector(const Matrix_d & matrix)
{ if (matrix.size() != matrix[0].size())
        {   
            std::cout << "Matrix is not square" << std::endl;
            return {};
        }
        Linear_Algebra lin;
        vector_d v(matrix.size());
        Matrix_d W = lin.Diagonalize(matrix);
        for(int i = 0 ; i < matrix.size(); i++)
        {
            v[i] =  -W[i][i];
        }
        return v;
    }

void Linear_Algebra_GPU::CudaT(const float *a, float *b, int m, int n)
    {
        float *M_a, *M_b;
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaMalloc((void **)&M_a, m * n * sizeof(float));
        cudaMalloc((void **)&M_b, m * n * sizeof(float));

        cudaMemcpyAsync(M_a, a, m * n * sizeof(float), cudaMemcpyHostToDevice, stream);
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

        tkernel<<<gridDim, blockDim, 0, stream>>>(M_a, M_b, m, n);
        cudaMemcpyAsync(b, M_b, m * n * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaFree(M_a);
        cudaFree(M_b);
        cudaStreamDestroy(stream);


    }

void Linear_Algebra_GPU::CudaMM(float *a, float *b, float *c, int m, int n, int p, int stream_id)
    {
        float *M_a, *M_b, *M_c;
        cudaStream_t stream = get_stream(stream_id);

        cudaMalloc((void **)&M_a, m * n * sizeof(float));
        cudaMalloc((void **)&M_b, n * p * sizeof(float));
        cudaMalloc((void **)&M_c, m * p * sizeof(float));

        cudaMemcpyAsync(M_a, a, m * n * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(M_b, b, n * p * sizeof(float), cudaMemcpyHostToDevice, stream);

        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((p + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

        mmkernel<<<gridDim, blockDim, 0, stream>>>(M_a, M_b, M_c, m, n, p);

        cudaMemcpyAsync(c, M_c, m * p * sizeof(float), cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        cudaFree(M_a);
        cudaFree(M_b);
        cudaFree(M_c);
    }

void Linear_Algebra_GPU::printVector(const vector_d vector)
    {
        for (float val: vector)
        {
            std::cout<< val << " "; 
        }
        std::cout << " " << std::endl;
        std::cout << "---------------------------\n";
    }

// Function to print the matrix 
void Linear_Algebra_GPU::printMatrix(const Matrix_d& matrix) 
    {
        for (const auto& row : matrix) 
        {
            for (float val : row) 
            {
                std::cout << val << "\t";  // Print each value with a tab space
            }
        std::cout << std::endl;  // Newline after each row
        }
        std::cout << "-----------------------------------\n";  // Separator for clarity

    }

void Linear_Algebra_GPU::printMatrixHead(const Matrix_d& matrix) 
    {   int row = matrix.size();
        int col = matrix[0].size();

        for (int i =0; i < std::min(5, row); i++)
        {
            for (int j = 0; j < std::min(5, col); j++)
            {
                std::cout << matrix[i][j] << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << " ----------------------------------------\n"; // Seperate matrix for clarity
        

    }

Matrix_d Linear_Algebra_GPU::Transpose(const Matrix_d & matrixA)
    {
        int m = matrixA.size();
        int n = matrixA[0].size();
        float *a = (float *)malloc(m * n * sizeof(float));
        float *b = (float *)malloc(m * n * sizeof(float));
        for (int i = 0; i < m; i++)
        {
            for(int j =0; j < n; j++)
            {
                a[i * n + j] = matrixA[i][j];
            }
        }
        Linear_Algebra_GPU lin;
        lin.CudaT(a, b, m, n);

        free(a);
        Matrix_d MatrixB(n, vector_d(m));
        for (int i = 0; i < n; i++)
        {
            for(int j =0; j < m; j++)
            {
                MatrixB[i][j] = b[i * m + j];
            }
        }
        free(b);

    
    return MatrixB;

    }
    
Matrix_d Linear_Algebra_GPU::MatMul(const Matrix_d & matrixA, const Matrix_d & matrixB, int stream_id)
    {
        int m = matrixA.size();
        int n = matrixA[0].size();
        int p = matrixB[0].size();

        if(n != matrixB.size())
        {   
            std::cout << "Size mismatch in matrix multiplication : (" << m << " x "<< n <<")" <<"," << "(" << matrixB.size() << " x "<< p <<")" << std::endl;
        }

        float *a = (float *)malloc(m * n * sizeof(float));
        float *b = (float *)malloc(n * p * sizeof(float));
        float *c = (float *)malloc(m * p * sizeof(float));


        for (int i = 0; i < m; i++)
        {
            for(int j =0; j < n; j++)
            {
                a[i * n + j] = matrixA[i][j];
            }
        }

        for (int i = 0; i < n; i++)
        {
            for(int j =0; j < p; j++)
            {
                b[i * p + j] = matrixB[i][j];
            }
        }


        this->CudaMM(a, b, c, m, n, p, stream_id);

        free(a);
        free(b);

        Matrix_d MatrixC(m, vector_d(p));
        for (int i = 0; i < m; i++)
        {
            for(int j =0; j < p; j++)
            {
                MatrixC[i][j] = c[i * p + j];
            }
        }
        free(c);

    
    return MatrixC;
    }

Linear_Algebra_GPU::Linear_Algebra_GPU() : num_heads(0) 
{
    InitializeStreams(10);
}

size_t Linear_Algebra_GPU::stream_count()
{
    return streams.size();
}

Linear_Algebra_GPU::~Linear_Algebra_GPU() 
{
    for (auto& stream : streams) {
        cudaStreamDestroy(stream);
    }
}

// Add these methods for stream management
void Linear_Algebra_GPU::InitializeStreams(int num_streams) 
{
    this->num_heads = num_streams;
    streams.resize(num_streams);
    for (int i = 0; i < num_streams; ++i) 
    {
        cudaStreamCreate(&streams[i]);
    }
}

cudaStream_t Linear_Algebra_GPU::get_stream(int stream_id)
{ 
    if (stream_id < 0 || stream_id >= streams.size()) 
    {   
        std::cout << "Invalid stream_id of id: " << stream_id <<std::endl;
        throw std::out_of_range("Invalid stream_id");
    }
    return streams[stream_id]; 
}

vector_d Math::random_vector(const int& size, float scale)
{   
    vector_d result(size);
    // Use random device to seed the generator
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister engine

        // Uniform distribution in range (-2, 2)
        std::uniform_real_distribution<> dist(-scale, scale);

        // Fill the vector with random values
        for (int i = 0; i < size; ++i) 
        {
            result[i] = dist(gen);
        }
    return result;
}

int Math::random_int(const int &a, const int& b)
{
    int result;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(a, b);
    result = dist(gen);
    return result;
}

float Math::round(float X, int decimal)
{
    if (X != X){throw std::invalid_argument("Value of round input is nan \n");}
    int dec = power(10, decimal);
    return std::round(dec * X) / dec;

}

float Math::sqrt(const float & x, int approx)
{   
    if (x != x){throw std::invalid_argument("Value of sine is nan \n");}
    
    if(x < 0) {throw std::invalid_argument("Sqrt input is negative \n");}
    if(x < 1e-40) {return x;}
    float t = x;
    for(int i = 0; i < approx; i++)
    {
        t = 0.5 * (t + (x/t));
    }
    if (t != t)
    {
        throw std::invalid_argument("Value of sqrt is nan \n");
    }
    return t;
}

float Math::max(const vector_d &vector)
    {
        float value = vector[0];
        for(int i = 1; i < vector.size(); i++)
        {   
            if (vector[i] > value)
            {
                value = vector[i];
            }
        }
        return value;
    }

float Math::min(const vector_d & vector)
    {
        float value = vector[0];
        for(int i = 1; i < vector.size(); i++)
        {   
            if (vector[i] < value)
            {
                value = vector[i];
            }
        }
        return value;
    }

float Math::ln(float n, int approx) // O(ln(n)) time complexity
{   
    if (n != n) {throw std::invalid_argument("Value of logarithm \n");}

    if(n == 1)
    {
        return 0;
    }

    if (n == 0)
    {
    throw std::invalid_argument("ln(0) is undefined \n");
    }  

    if( n < 1e-2)
    {   
        return -ln(1 / n, approx);
    }
        
    float e = 2.71828182846;
    float t = 0;
    while (n > e || (n / e >= 0.825 && n / e < 2))
    {
        ++t;
        e *= 2.71828182846;
    }
        return t + 1 + ln_sub1(n / e, approx);
}
    
long long Math::factorial_integer(int value) // This factorial algorithm only works for integers O(n)
    {   
        if (value != value)
        {
        throw std::invalid_argument("Value of sqrt is nan \n");
        }

        if (value <= 1)
        {
            return 1;
        }
        else
        {
            return value * factorial_integer(value - 1);
        }

    }

float Math::power(float x, int n)  // Function is defined for all integers O(log(n))
    {   if (n % 1 != 0)
        {
            std::cout << " This function only operates on integer values" << std::endl;
        }
        if (n < 0)
        {
            return 1 / power(x, -n);
        }
        if (n==0)
        {
            return 1;
        }
        else if(n % 2 == 0)
        {
            float half = power(x, n /2);
            return half * half;
        }
        else
        {
            return x * power(x, n - 1);
        }
    }

float Math::power_fractional(float x, float n) // Function is defined for all numbers to all powers .5f accuracy
    {   
        if (x != x){throw std::invalid_argument("Value of sqrt is nan \n");}

        if (n < 0)
        {
            return 1 / power_fractional(x, -n);
        }
        if (n==0)
        {
            return 1;
        }

        if (n== 1)
        {
            return x;
        }
        Math m;
        return m.exp(n * m.ln(x));
    }

float Math::exp(float x, int expansion_term) // O(1) time complexity
    {   
        if (x != x){throw std::invalid_argument("Value of exponent is nan \n");}

        float first_10_n = 1 + x + (power(x,2) / 2) + (power(x,3) / 6) + (power(x,4) / 24) + (power(x,5) / 120) + 
        (power(x,6) / 720) +(power(x,7) / 5040) + (power(x,8) / 40320) +(power(x,9) / 362880) +(power(x,10) / 3628800);

        if (expansion_term <= 10)
        {
            return first_10_n;
        }
        else
        {
            for(int i=11; i < expansion_term + 1; i++)
            {
                first_10_n += power(x,i) / factorial_integer(i);
            }
            return first_10_n;
        }
        
    }

float Math::combinatorics_choose(float x, int k) // Works fine for X's and integer K's
    {
        float value = x;
        for(int i=1; i < k; i++)
        {
            x *= value - i;  
        }
        return  x  / factorial_integer(k);
    }

float Math::sin(float y) // O(1) time complexity
{   
    if (y != y){throw std::invalid_argument("Value of sine is nan \n");}
    Math m;
    float pi = 3.1415927;
    int t = y / pi;
    float x = y - t * pi;
    float series = x - (m.power(x,3)/6) + (m.power(x,5)/120) - (m.power(x,7)/5040) + (m.power(x,9)/362880) - (m.power(x,11)/39916800);
    if(t % 2 == 0)
    {
        return series;
    }
    else
    {
        return -series;
    }
}

float Math::cos(float y) // O(1) time complexity
{   
    if (y != y){throw std::invalid_argument("Value of cosine is nan \n");}
    Math m;
    float pi = 3.1415927;
    int t = y / pi;
    float x = y - t * pi;
    float series = 1 - (m.power(x,2)/2) + (m.power(x,4)/24) - (m.power(x,6)/720) + (m.power(x,8)/40320) - (m.power(x,10)/362880);
    if(t % 2 == 0)
    {
        return series;
    }
    else
    {
        return -series;
    }
}

float Math::ln_sub1(float y, int approx) // O(1) time complexity
{   
    if (y != y){throw std::invalid_argument("Value of sine is nan \n");}

    float x = y - 1;
    if (abs(x) >= 1)
    {
        throw std::invalid_argument("Outside boundary of convergence \n");
    }
    else
    {
        float sum  = 0.0;
        for(int i = 1; i < approx; i++)
        {
            if(i% 2 == 1)
            {
                sum += power(x, i) / i;
            }
            else
            {
                sum -= power(x,i) / i;
            }
        }
        return sum;
    }
}

int Math::TopKSampler(const vector_d& prob, const int &k)
{   
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, 1 - 1e-7);


    int size  = prob.size();
    prob_id pair_id(size, {0.0f, 0});
    for(int i = 0; i < size; i++){pair_id[i] = {prob[i], i};}

    // Sort by first value
    std::sort(pair_id.begin(), pair_id.end(), [](const auto& a, const auto& b){return a.first > b.first;});

    prob_id elem(k, {0.0f, 0});
    float sum = 0;
    for(int i = 0; i < k; i++)
    {sum += pair_id[i].first;
     elem[i] = pair_id[i];}

    float result;
    result = dist(gen) + 1e-7;

    for(int i = 0; i < k; i++)
    {   
        elem[i].first = round(elem[i].first / sum, 3);  
    }
    
    int T = 0;

    while(result > 0)
    {
        result -= elem[T].first;
        T++;
    }

    return elem[T-1].second;
}

vector_d Linear_Algebra::vectorsum(const vector_d& v1, const vector_d & v2)
{   
    int size = v1.size();
    if(size != v2.size())
    {
        std::cout << "Cannot add vector length: " << v1.size() << " + " << v2.size() << std::endl;
        return {};
    }
    else
    {   
        vector_d v3(size, 0.0f);
        for(int i = 0; i  < v1.size(); i++)
        {
            v3[i] = v1[i] + v2[i];
        }
        return v3;
    }
}

float Machine_Learning::max(vector_d vector)
    {
        float value = vector[0];
        for(int i = 1; i < vector.size(); i++)
        {   
            if (vector[i] > value)
            {
                value = vector[i];
            }
        }
        return value;
    }

float Machine_Learning::min(vector_d vector)
    {
        float value = vector[0];
        for(int i = 1; i < vector.size(); i++)
        {   
            if (vector[i] < value)
            {
                value = vector[i];
            }
        }
        return value;
    }

long long Machine_Learning::factorial_integer(int value) // This factorial algorithm only works for integers O(n)
    {
        if (value <= 1)
        {
            return 1;
        }
        else
        {
            return value * factorial_integer(value - 1);
        }

    }

float Machine_Learning::exp(float x, int expansion_term)
    {
        float first_10_n = 1 + x + (power(x,2) / 2) + (power(x,3) / 6) + (power(x,4) / 24) + (power(x,5) / 120) + 
        (power(x,6) / 720) +(power(x,7) / 5040) + (power(x,8) / 40320) +(power(x,9) / 362880) +(power(x,10) / 3628800);

        if (expansion_term <= 10)
        {
            return first_10_n;
        }

        else
        {
            for(int i=11; i < expansion_term + 1; i++)
            {
                first_10_n += power(x,i) / factorial_integer(i);
            }
            return first_10_n;
        }
        
    }

// Fixed derivative function  // O(n)
std::function<float(float)>  Machine_Learning::derivative(std::function<float(float)> function,int order, float h)
        {
            // Helper function for recursive derivative calculation
            std::function<std::function<float(float)>(std::function<float(float)>, int, float)> impl =
                [&impl](std::function<float(float)> f, int o, float step) -> std::function<float(float)> {
                    return [=](float x) 
                    {
                        if (o == 1) 
                        {
                            return (f(x + step) - f(x)) / step;
                        } 
                        else 
                        {
                            auto lower_derivative = impl(f, o - 1, step);
                            return (lower_derivative(x + step) - lower_derivative(x)) / step;
                        }
                    };
                };
            
            return impl(function, order, h);
        }
        
float Machine_Learning::power(float x, int n)  // Function is defined for all integers O(log(n))
        {   if (n % 1 != 0)
        {
            std::cout << " This function only operates on integer values" << std::endl;
        }
        if (n < 0)
        {
            return 1 / power(x, -n);
        }
        if (n==0)
        {
            return 1;
        }
        else if(n % 2 == 0)
        {
            float half = power(x, n /2);
            return half * half;
        }
        else
        {
            return x * power(x, n - 1);
        }
        }
        
std::function<float(float)> Machine_Learning::loss(std::function<float(float)> function)
        {
            return [this, function](float x) {
                Machine_Learning ml;
                return ml.power(function(x), 2);
            };
        }
        
vector_d Machine_Learning::single_training( std::function<float(float)> function, int epochs, float lr)
        {   
            Math ma;
            vector_d param = {1};
            float B1 = 0.9f, B2 = 0.999f, epsilon = 1e-7f;
            float m = 0.0f, v = 0.0f;
            for(int epoch = 0; epoch < epochs; epoch++)
            {   
                float gradient = derivative(loss(function), 1)(param[0]);
                m = B1 * m + (1 - B1) * gradient;
                v = B2 * v + (1 - B2) * power(gradient, 2);
                float m_hat = m / (1 - ma.power(B1, epoch + 1));
                float v_hat = v / (1 - ma.power(B2, epoch + 1));
                param[0] -= lr * m_hat / (ma.sqrt(v_hat) + epsilon);
            }
            return param;
        }

Matrix_d Machine_Learning::RelU(Matrix_d matrix)
        {
            int row = matrix.size(), col = matrix[0].size();
            for(int i = 0; i < row; i++)
            {
                for(int j = 0; j < col; j++)
                {
                   if (matrix[i][j] < 0)
                   {
                    matrix[i][j] = 0;
                   }
                }
            }
            return matrix;
        }

Matrix_d Machine_Learning::deriv_ReLU(Matrix_d matrix)
        {
            int row = matrix.size(), col = matrix[0].size();
            for(int i = 0; i < row; i++)
            {
                for(int j = 0; j < col; j++)
                {
                    if(matrix[i][j] > 0)
                    {
                        matrix[i][j] = 1;
                    }
                    else
                    {
                        matrix[i][j] = 0;
                    }
                }
            }
            return matrix;
        }

vector_d Machine_Learning::softmax(vector_d Z)
        {
            int len = Z.size();
            float sum = 0;
            Machine_Learning ml;
            for (int i = 0; i < len; i++)
            {
               Z[i] =  ml.exp(Z[i]);
               sum +=Z[i];
            }

            for (int i = 0; i < len; i++)
            {

        
                Z[i] = Z[i] / sum;
            }
            return Z;
        }

vector_d Machine_Learning::deriv_softmax(vector_d Z)
        {
            int len = Z.size();
            float sum = 0;
            Machine_Learning ml;
            for (int i = 0; i < len; i++)
            {
               Z[i] =  ml.exp(Z[i]);
               sum +=Z[i];
            }

            for (int i = 0; i < len; i++)
            {

        
                Z[i] = (Z[i] / sum) *(1- Z[i] / sum);
            }
            return Z;
        }
        
Matrix_d Machine_Learning::SoftMax(Matrix_d Z)
        {   
            int row = Z.size();
            for(int i = 0; i < row; i++)
            {
                Z[i] = softmax(Z[i]);
            }
            return Z;
        }

Matrix_d Machine_Learning::deriv_SoftMax(Matrix_d Z)
        {   
            int row = Z.size();
            for(int i = 0; i < row; i++)
            {
                Z[i] = deriv_softmax(Z[i]);
            }
            return Z;
        }

float Linear_Algebra::mean(const vector_d & Z)
{   
    float sum = 0;
    int size = Z.size();
    for(int i = 0; i < size; i++)
    {
        sum += Z[i]/size;
    }
    return sum;
}

float Linear_Algebra::std(const vector_d & Z, const float & error)
{   
    Math m;
    float sum = 0;
    float avg = mean(Z);
    int size = Z.size();
    for(int i = 0; i < size; i++)
    {
        sum += (Z[i] - avg) * (Z[i] - avg) / size;
    }
    return m.sqrt(sum + error);

}

Matrix_d Linear_Algebra::LayerNorm(Matrix_d Z, float error)
{
    for(int i  = 0; i < Z.size(); i++)
    {
        float mean_val = mean(Z[i]);
        float standard = std(Z[i], error);

        for(int j = 0; j < Z[0].size(); j++)
        {
            Z[i][j] = (Z[i][j] - mean_val)/ standard;
        }
    }
    return Z;
}

vector_d Machine_Learning::softmask(vector_d Z, int t)
        {
            int len = Z.size();
            float sum = 0;
            Machine_Learning ml;
            for (int i = 0; i < len; i++)
            {  
                if(i < t)
                {
                Z[i] =  ml.exp(Z[i]);
                sum +=Z[i];
                }
                else
                {
                    Z[i] = 0;
                }

            }

            for (int i = 0; i < t; i++)
            {
                Z[i] = Z[i] / sum;
            }
            return Z;
        }

vector_d Machine_Learning::deriv_softmask(vector_d Z, int t)
        {
            int len = Z.size();
            float sum = 0;
            for (int i = 0; i < len; i++)
            {   
                if(i < t)
                {
                Z[i] =  exp(Z[i]);
                sum +=Z[i];
                }
                else
                {
                    Z[i] = 0;
                }

            }

            for (int i = 0; i < t; i++)
            {
                Z[i] = (Z[i] / sum) *(1- Z[i] / sum);
            }
            return Z;
        }
        
Matrix_d Machine_Learning::SoftMask(Matrix_d Z)
        {   

            int row = Z.size();
            for(int i = 0; i < row; i++)
            {
                Z[i] = softmask(Z[i], i + 1);
            }
            return Z;
        }

Matrix_d Machine_Learning::deriv_SoftMask(Matrix_d Z)
        {   
            int row = Z.size();
            for(int i = 0; i < row; i++)
            {
                Z[i] = deriv_softmask(Z[i], i + 1);
            }
            return Z;
        }





