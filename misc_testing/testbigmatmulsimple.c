#include <stdio.h>
#include <time.h>
#include <stdlib.h>

float* matrix_mul(float ** a, float* b, int n, int m) {
  float* output = (float*) calloc(n, sizeof(float));

  for(int r = 0; r < n; r++)
    for(int c = 0; c < m; c++)
      output[r] += a[r][c] * b[c];
  
  return output;
}


int main() {
  srand(time(NULL));
  
  int power = 15;
  int bigN = (1<<power), bigM = (1<<15);
  
  float** bigA = (float**)(malloc(sizeof(float*) * bigN));
  for(int i=0;i<bigN;i++)
    bigA[i] = (float*)(malloc(sizeof(float) * bigM));

  printf("successfully malloced a\n");
  
  float* bigB = (float*)(malloc((sizeof(float) * bigM)));
  
  printf("successfully malloced b\n");

  for(int i=0;i<bigN;i++) {
    for(int j=0;j<bigM;j++) {
      bigA[i][j] = rand()/(float)RAND_MAX;
      bigB[j] = rand()/(float)RAND_MAX;
    }
  }

  printf("successfully initialized\n");
  
  struct timeval stop, start;
  gettimeofday(&start, NULL);
  float* output = matrix_mul(bigA, bigB, bigN, bigM);
  gettimeofday(&stop, NULL);

  for(int i=0;i<bigN;i++)  printf("%f\n", output[i]);

  printf("took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec); 
}
