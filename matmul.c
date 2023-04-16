float** matmul(float ** a, float** b, int n, int m, int k) {
  // a is n*m and b is m*k, output is n*k 
  float ** output = (float**)malloc(sizeof(float*) * n);
  for(int i=0;i<n;i++) output[i] = (float*)(calloc(sizeof(float) * k));
  for(int r=0;r<n;r++)
    for(int c=0;c<k;c++)
      for(int j=0;j<m;j++)
	output[r][c] += a[r][j] * b[j][c];
  return output;
}


float** sigmoid(float** mat, int n, int m) {
  float ** output = (float**)malloc(sizeof(float*) * n);
  for(int i=0;i<n;i++) output[i] = (float*)(calloc(sizeof(float) * k));
  
  for(int i=0;i<n;i++)
    for(int j=0;j<m;j++)
      output[i][j] = 1.0/(1+exp(-mat[i][j]));

  return output;
}
