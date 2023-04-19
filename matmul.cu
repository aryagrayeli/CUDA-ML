#pragma once

#include <stdlib.h>
#include <math.h>

// Batching - Input vectors are an array of vectors

// Spawn N Threads
__global__ void matrix_mul(float * a, float * b, float * output, int N, int M) {
    // int n = gridDim.x;
    // int m = blockDim.x;
    // int r = blockIdx.x;
    // int c = threadIdx.x;

    // int Mi = 31 - __builtin_clz(m);
    // int M = 1 << Mi;
    // __shared__ float arr[M];
    // __shared__ float B[m];
    // cudaMemcpy(arr, a[r], sizeof(float) * m, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(B, b, sizeof(float) * m, cudaMemcpyDeviceToDevice);
	// cudaMemset(arr + m, 0, sizeof(float) * (M - m));
    // arr[c] *= B[c];
    // __syncthreads();
    // for (int i = 0; i < Mi; i++) {
    //     if (c >> (Mi - i - 1)) {
    //         continue;
    //     }
    //     int a = c << (i + 1);
    //     int b = a + (1 << i);
    //     arr[a] += arr[b];
    //     __syncthreads();
    // }

    // output[r] = *arr; 
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < N) {
        float sum = 0;
        for(int i = 0; i < M; i++) 
            sum += a[row * M + i] * b[i];
        output[row] = sum;
    }
}

// Spawn B Threads
__global__ void batch_matrix_mul(float * a, float * b, float * output, int N, int M, int B) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if(batch < B) {
        dim3 gridSz(1, 1, 1);
        dim3 blockSz(N, M, 1);
        matrix_mul<<<gridSz, blockSz>>>(a, b + (batch * M), output + (batch * N), N, M);
    }
}

// Spawn N Threads
__global__ void vector_sigmoid(float * input, float * output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
        output[i] = 1.0 / (1 + exp(-input[i]));
}

// Spawn B Threads
__global__ void batch_vector_sigmoid(float * input, float * output, int N, int B) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if(batch < B) {
        dim3 gridSz(1, 1, 1);
        dim3 blockSz(N, 1, 1);
        vector_sigmoid<<<gridSz, blockSz>>>(input + (batch * N), output + (batch * N), N);
    }
}

// Spawn N Threads
__global__ void vector_dsigmoid(float * input, float * output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        float sigmoid = 1.0 / (1 + exp(-input[i]));
        output[i] = sigmoid * (1-sigmoid);
    }
}

// Spawn B Threads
__global__ void batch_vector_dsigmoid(float * input, float * output, int N, int B) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if(batch < B) {
        dim3 gridSz(1, 1, 1);
        dim3 blockSz(N, 1, 1);
        vector_dsigmoid<<<gridSz, blockSz>>>(input + (batch * N), output + (batch * N), N);
    }
}

// Spawn N Threads
__global__ void vector_relu(float * input, float * output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
        output[i] = input[i] > 0 ? input[i] : 0;
}

// Spawn B Threads
__global__ void batch_vector_relu(float * input, float * output, int N, int B) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if(batch < B) {
        dim3 gridSz(1, 1, 1);
        dim3 blockSz(N, 1, 1);
        vector_relu<<<gridSz, blockSz>>>(input + (batch * N), output + (batch * N), N);
    }
}

// Spawn N Threads
__global__ void vector_drelu(float * input, float * output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
        output[i] = input[i] > 0;
}

// Spawn B Threads
__global__ void batch_vector_drelu(float * input, float * output, int N, int B) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if(batch < B) {
        dim3 gridSz(1, 1, 1);
        dim3 blockSz(N, 1, 1);
        vector_drelu<<<gridSz, blockSz>>>(input + (batch * N), output + (batch * N), N);
    }
}

// Spawn N Threads
__global__ void vector_tanh(float * input, float * output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
        output[i] = (exp(input[i]) - exp(-input[i])) / (exp(input[i]) + exp(-input[i]));
}

// Spawn B Threads
__global__ void batch_vector_tanh(float * input, float * output, int N, int B) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if(batch < B) {
        dim3 gridSz(1, 1, 1);
        dim3 blockSz(N, 1, 1);
        vector_tanh<<<gridSz, blockSz>>>(input + (batch * N), output + (batch * N), N);
    }
}

// Spawn N Threads
__global__ void vector_dtanh(float * input, float * output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        float tanh = (exp(input[i]) - exp(-input[i])) / (exp(input[i]) + exp(-input[i]));
        output[i] = 1 - tanh*tanh;
    }
}

// Spawn B Threads
__global__ void batch_vector_dtanh(float * input, float * output, int N, int B) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if(batch < B) {
        dim3 gridSz(1, 1, 1);
        dim3 blockSz(N, 1, 1);
        vector_dtanh<<<gridSz, blockSz>>>(input + (batch * N), output + (batch * N), N);
    }
}

// Spawn N Threads
__global__ void vector_softmax(float * input, float * output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        float sum = 0;
        for(int k = 0; k < N; k++)
            sum += exp(input[k]);
        output[i] = exp(input[i])/sum;
    }
}

// Spawn B Threads
__global__ void batch_vector_softmax(float * input, float * output, int N, int B) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if(batch < B) {
        dim3 gridSz(1, 1, 1);
        dim3 blockSz(N, 1, 1);
        vector_softmax<<<gridSz, blockSz>>>(input + (batch * N), output + (batch * N), N);
    }
}

// Spawn N Thread
__global__ void vector_dsoftmax(float* input, float * output, int j, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        float sum = 0;
        for(int k = 0; k < N; k++)
            sum += exp(input[k]);
        
        float sj = exp(input[j])/sum;
        if(i == j)
            output[i] = sj * (1-sj);
        else
            output[i] = -sj*(exp(input[i])/sum);
    }
}

// Spawn B Threads
__global__ void batch_vector_dsoftmax(float * input, float * output, int * js, int N, int B) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if(batch < B) {
        dim3 gridSz(1, 1, 1);
        dim3 blockSz(N, 1, 1);
        vector_dsoftmax<<<gridSz, blockSz>>>(input + (batch * N), output + (batch * N), js[batch], N);
    }
}

// Spawn N Threads
__global__ void vector_add(float * a, float * b, float * output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
        output[i] = a[i] + b[i];
}

// Spawn B Threads
__global__ void batch_vector_add(float * a, float * b, float * output, int N, int B) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if(batch < B) {
        dim3 gridSz(1, 1, 1);
        dim3 blockSz(N, 1, 1);
        vector_add<<<gridSz, blockSz>>>(a + (batch * N), b, output + (batch * N), N);
    }
}

// Spawn N Threads
__global__ void matrix_hadamard(float * a, float * b, float * output, int N, int M) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < N && col < M) 
        output[row * M + col] = a[row * M + col] * b[row * M + col];
}

// Spawn N x M Threads
__global__ void matrix_trans(float * input, float * output, int N, int M) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < N && col < M) 
        output[col * N + row] = input[row * M + col];
}

// Spawn N x M Threads
__global__ void matrix_scalar(float * input, int sc, float * output, int N, int M) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < N && col < M) 
        output[row * M + col] = sc * input[row * M + col];
}