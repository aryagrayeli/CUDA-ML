#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "matmul.h"

int main() {
    int n = 3, m = 3;

    float** A = (float**) malloc(sizeof(float*) * n);
    for(int i = 0; i < n; i++) A[i] = (float*) malloc(sizeof(float) * m);

    // [1 -2 3]
    // [4 5 6]
    // [-7 8 9]

    A[0][0] = 1; A[0][1] = -2; A[0][2] = 3; A[1][0] = 4; A[1][1] = 5; A[1][2] = 6; A[2][0] = -7; A[2][1] = 8; A[2][2] = 9;

    float** B = (float**) malloc(sizeof(float*) * n);
    for(int i = 0; i < n; i++) B[i] = (float*) malloc(sizeof(float) * m);

    // [-1 2 3]
    // [5 -5 2]
    // [-1 8 4]

    B[0][0] = -1; B[0][1] = 2; B[0][2] = 3; B[1][0] = 5; B[1][1] = -5; B[1][2] = 2; B[2][0] = -1; B[2][1] = 8; B[2][2] = 4;

    float* output = matrix_mul(A, B[0], 3, 3);
    printf("Test Matrix Multiplication\n");
    printf("Output: %f, %f, %f; Should be: 4, 24, 50\n\n", output[0], output[1], output[2]);

    output = vector_sigmoid(A[0], 3);
    printf("Test sigmoid function\n");
    printf("Output: %f, %f, %f; Should be: 0.731058, 0.119202, 0.952574\n\n", output[0], output[1], output[2]);

    output = vector_dsigmoid(A[0], 3);
    printf("Test sigmoid function derivative\n");
    printf("Output: %f, %f, %f; Should be: 0.196611, 0.104993, 0.045176\n\n", output[0], output[1], output[2]);

    output = vector_relu(A[0], 3);
    printf("Test relu function\n");
    printf("Output: %f, %f, %f; Should be: 1, 0, 3\n\n", output[0], output[1], output[2]);

    output = vector_drelu(A[0], 3);
    printf("Test relu function derivative\n");
    printf("Output: %f, %f, %f; Should be: 1, 0, 1\n\n", output[0], output[1], output[2]);

    output = vector_tanh(A[0], 3);
    printf("Test tanh function\n");
    printf("Output: %f, %f, %f; Should be: 0.761594, -0.964027, 0.995054\n\n", output[0], output[1], output[2]);

    output = vector_dtanh(A[0], 3);
    printf("Test tanh function derivative\n");
    printf("Output: %f, %f, %f; Should be: 0.419974, 0.070650, 0.009866\n\n", output[0], output[1], output[2]);

    output = vector_softmax(A[0], 3);
    printf("Test softmax function\n");
    printf("Output: %f, %f, %f; Should be: 0.118499, 0.005899, 0.875600\n\n", output[0], output[1], output[2]);

    output = vector_dsoftmax(A[0], 0, 3);
    printf("Test softmax function derivative with j = 0\n");
    printf("Output: %f, %f, %f; Should be: 0.104457, −0.000699, −0.103758\n\n", output[0], output[1], output[2]);

    output = vector_dsoftmax(A[0], 1, 3);
    printf("Test softmax function derivative with j = 1\n");
    printf("Output: %f, %f, %f; Should be: −0.000699, 0.005864, −0.005165\n\n", output[0], output[1], output[2]);

    output = vector_dsoftmax(A[0], 2, 3);
    printf("Test softmax function derivative with j = 2\n");
    printf("Output: %f, %f, %f; Should be: −0.103758, −0.005165, 0.108924\n\n", output[0], output[1], output[2]);

    output = vector_add(A[0], B[0], 3);
    printf("Test vector addition\n");
    printf("Output: %f, %f, %f; Should be: 0, 0, 6\n\n", output[0], output[1], output[2]);

    float** output2D = matrix_hadamard(A, B, 3, 3);
    printf("Test matrix with hadamard transformation\n");
    printf("Output: \n%f, %f, %f\n%f, %f, %f\n%f, %f, %f;\nShould be: \n-1, -4, 9\n20, -25, 12\n7, 64, 36\n\n", output2D[0][0], output2D[0][1], output2D[0][2], output2D[1][0], output2D[1][1], output2D[1][2], output2D[2][0], output2D[2][1], output2D[2][2]);

    output2D = matrix_trans(A, 3, 3);
    printf("Test matrix transposition\n");
    printf("Output: \n%f, %f, %f\n%f, %f, %f\n%f, %f, %f;\nShould be: \n1, 4, -7\n-2, 5, 8\n3, 6, 9\n\n", output2D[0][0], output2D[0][1], output2D[0][2], output2D[1][0], output2D[1][1], output2D[1][2], output2D[2][0], output2D[2][1], output2D[2][2]);

    output2D = matrix_scalar(A, 10, 3, 3);
    printf("Test matrix multiplied with a scalar\n");
    printf("Output: \n%f, %f, %f\n%f, %f, %f\n%f, %f, %f;\nShould be: \n10, -20, 30\n40, 50, 60\n-70, 80, 90\n\n", output2D[0][0], output2D[0][1], output2D[0][2], output2D[1][0], output2D[1][1], output2D[1][2], output2D[2][0], output2D[2][1], output2D[2][2]);
}