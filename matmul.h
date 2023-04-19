#pragma once

#include <stdlib.h>
#include <math.h>

float* matrix_mul(float ** a, float* b, int n, int m) {
  float* output = (float*) calloc(n, sizeof(float));

  for(int r = 0; r < n; r++)
    for(int c = 0; c < m; c++)
	    output[r] += a[r][c] * b[c];
  
  return output;
}

float* vector_sigmoid(float* input, int n) {
  float* output = (float*) malloc(sizeof(float) * n);
  
  for(int i = 0; i < n; i++)
    output[i] = 1.0 / (1 + exp(-input[i]));

  return output;
}

float* vector_dsigmoid(float* input, int n) {
  float* output = (float*) malloc(sizeof(float) * n);
  
  for(int i = 0; i < n; i++) {
    float sigmoid = 1.0 / (1 + exp(-input[i]));
    output[i] = sigmoid * (1-sigmoid);
  }
  return output;
}

float* vector_relu(float* input, int n) {
  float* output = (float*) malloc(sizeof(float) * n);

  for(int i = 0; i < n; i++)
    output[i] = input[i] > 0 ? input[i] : 0;
        
  return output;
}

float* vector_drelu(float* input, int n) {
  float* output = (float*) malloc(sizeof(float) * n);

  for(int i = 0; i < n; i++)
    output[i] = input[i] > 0;

  return output;
}

float* vector_tanh(float* input, int n) {
  float* output = (float*) malloc(sizeof(float) * n);

  for(int i = 0; i < n; i++)
    output[i] = (exp(input[i]) - exp(-input[i])) / (exp(input[i]) + exp(-input[i]));

  return output;
}

float* vector_dtanh(float* input, int n) {
  float* output = (float*) malloc(sizeof(float) * n);

  for(int i = 0; i < n; i++) {
    float tanh = (exp(input[i]) - exp(-input[i])) / (exp(input[i]) + exp(-input[i]));
    output[i] = 1 - tanh*tanh;
  }

  return output;
}

float* vector_softmax(float* input, int n) {
  float* output = (float*) malloc(sizeof(float) * n);

  float sum = 0;
  for(int i = 0; i < n; i++)
    sum += exp(input[i]);
  for(int i = 0; i < n; i++)
    output[i] = exp(input[i])/sum;
  
  return output;
}

float* vector_dsoftmax(float* input, int j, int n) {
  float* output = (float*) malloc(sizeof(float) * n);

  float sum = 0;
  for(int i = 0; i < n; i++)
    sum += exp(input[i]);
  float sj = exp(input[j])/sum;
  for(int i = 0; i < n; i++)
    if(i == j)
      output[i] = sj * (1-sj);
    else
      output[i] = -sj*(exp(input[i])/sum);
  
  return output;
}

float* vector_add(float* a, float* b, int n) {
  float* output = (float*) malloc(sizeof(float) * n);

  for(int i = 0; i < n; i++)
    output[i] = a[i] + b[i];

  return output;
}

float** matrix_hadamard(float** a, float** b, int n, int m) {
  float** output = (float**) malloc(sizeof(float*) * n);
  for(int i = 0; i < n; i++) output[i] = (float*) malloc(sizeof(float) * m);

  for(int r = 0; r < n; r++)
    for(int c = 0; c < m; c++)
      output[r][c] = a[r][c] * b[r][c];

  return output;
}

float** matrix_trans(float** input, int n, int m) {
  float** output = (float**) malloc(sizeof(float*) * m);
  for(int i = 0; i < n; i++) output[i] = (float*) malloc(sizeof(float) * n);

  for(int r = 0; r < m; r++)
    for(int c = 0; c < n; c++)
      output[r][c] = input[c][r];

  return output;
}

float** matrix_scalar(float** input, int c, int n, int m) {
  float** output = (float**) malloc(sizeof(float*) * n);
  for(int i = 0; i < n; i++) output[i] = (float*) malloc(sizeof(float) * m);

  for(int r = 0; r < n; r++)
    for(int c = 0; c < m; c++)
      output[r][c] = c * input[r][c];

  return output;
}