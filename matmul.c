float** matrix_mul(float ** a, float* b, int n, int m) {
  float* output = (float*) calloc(n, sizeof(float));

  for(int r = 0; r < n; r++)
    for(int c = 0; c < m; c++)
	    output[r] += a[r][c] * b[c];
  
  return output;
}


float* sigmoid(float* input, int n) {
  float* output = (float*) malloc(sizeof(float) * n);
  
  for(int i = 0; i < n; i++)
    output[i] = 1.0/(1+exp(-input[i]));

  return output;
}

float* relu(float* input, int n) {
  float* output = (float*) malloc(sizeof(float) * n);

  for(int i = 0; i < n; i++)
    output[i] = input[i] > 0 : input[i] ? 0;
        
  return output;
}

float* tanh(float* input, int n) {
  float* output = (float*) malloc(sizeof(float) * n);

  for(int i = 0; i < n; i++)
    output[i] = (exp(input[i]) - exp(-input[i])) / (exp(input[i]) + exp(-input[i]));

  return output;
}

float* softmax(float* input, int n) {
  float* output = (float*) malloc(sizeof(float) * n);

  float sum = 0;
  for(int i = 0; i < n; i++)
    sum += exp(input[i]);
  for(int i = 0; i < n; i++)
    output[i] = exp(input[i])/sum;
  
  return output;
}

float* vector_add(float* a, float* b, int n) {
  float* output = (float*) malloc(sizeof(float) * n);

  for(int i = 0; i < n; i++)
    output[i] = a[i] + b[i];

  return output;
}

float** hadamard_mul(float** a, float** b, int n, int m) {
  float** output = (float**) malloc(sizeof(float*) * n);
  for(int i = 0; i < n; i++) output[i] = malloc(sizeof(float) * m);

  for(int r = 0; r < n; r++)
    for(int c = 0; c < m; c++)
      output[r][c] = a[r][c] * b[r][c];

  return output;
}

float** matrix_trans(float** input, int n, int m) {
  float** output = (float**) malloc(sizeof(float*) * m);
  for(int i = 0; i < n; i++) output[i] = malloc(sizeof(float) * n);

  for(int r = 0; r < m; r++)
    for(int c = 0; c < n; c++)
      output[r][c] = input[c][r];

  return output;
}

float** matrix_scalar(float** input, int c, int n, int m) {
  float** output = (float**) malloc(sizeof(float*) * n);
  for(int i = 0; i < n; i++) output[i] = malloc(sizeof(float) * m);

  for(int r = 0; r < n; r++)
    for(int c = 0; c < m; c++)
      output[r][c] = c * input[r][c];

  return output;
}

// Derivative of Activations