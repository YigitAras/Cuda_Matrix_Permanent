#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>
#include <omp.h>
#include <stdlib.h>
#include <math.h>

using namespace std;
__device__ int getGlobIdx_1D_2D(){
    return blockIdx.x*blockDim.x*blockDim.y
                     +threadIdx.y * blockDim.x + threadIdx.x;
}
__global__ void preprocess(double * d_XX,double *d_mat,unsigned long long int SIZE,int N,unsigned long long int CHUNKY){
    unsigned long long int ind = threadIdx.x + blockIdx.x * blockDim.x;
    //unsigned long long int ind = getGlobIdx_1D_2D();
    unsigned long long int start_loc = ind*CHUNKY;
    unsigned long long int start_of_XX = ind*N;
    for(int i=0;i<N;i++){
      // since all arrays are flattened
      d_XX[start_of_XX + i] = d_mat[N*(N-1)+ i];
      for(int j=0; j<N; j++){
        d_XX[start_of_XX + i] -= ((double)d_mat[(j*N)+i]/2);
      }
    }

    unsigned long long int y = (start_loc>>1) ^ start_loc;
    for(int i=0;i<N;i++){
      for(int k=0;k<N;k++){
        if( ( (y >> k ) & 1LL ) == 1){
          d_XX[start_of_XX+i] += d_mat[N*k+i];  //    M[i][k]
      }
    }
  }
}



__global__ void perm_kernel(double * d_XX,unsigned long long int CHUNKY,double *d_p,double *d_mat,unsigned long long int SIZE,int N,unsigned long long int THREADS,unsigned long long int BLOCKS){
  unsigned long long int ind = threadIdx.x + blockIdx.x * blockDim.x;
  //unsigned long long int ind = getGlobIdx_1D_2D();
  unsigned long long int start_loc = ind*CHUNKY + 1;
  // GOTTO start from start_loc+1 then go until CHUNKY+1
  unsigned long long int LIMITER = start_loc+CHUNKY;
  // carefull last chunk start_loc might go one over...
  int ps = (start_loc & 1LL) == 0 ? -1:1;

  // do the calculations for the whole sha-bang
  double local_p = 0.0; // this for local, then reduce it to outer
  for(unsigned long long i = start_loc; (i < LIMITER) && (i < SIZE) ;i++){
      unsigned long long int y = (i>>1LL) ^ i; // gray code
      unsigned long long int yy = ( (i-1)>>1LL ) ^ (i-1); // i-1's gray-code
      long long int z = __ffsll( y ^ yy )-1;  // get the changing bit
      long long int s = ((y >> z)  & 1LL) == 1 ?  1:-1; // find changing bit

      double dd = 1.0;
      for(int j=0;j<N;j++){
        d_XX[(j*THREADS*BLOCKS)+ind] += s * d_mat[N*z+j]; // M[j][Z]
        dd *= d_XX[(j*THREADS*BLOCKS)+ind];
      }

      ps *= -1;
      local_p += ps * dd;
  }
  // do a reduction on the d_p !!!!
  atomicAdd(d_p,local_p);
}


void usage()
{
  cout << "USAGE: ./exec <filename> <machine no>" << endl;
  exit(0);
}

int main(int argc, const char** argv)
{

  if(argc != 3)
    usage();

  string line;

  const char* filename = argv[1];
  int MACHINE_NO = atoi(argv[2]);
  ifstream input (filename);
  if(input.fail())
    return 0;


  int N;
  int **M;
  getline(input,line);
  N = atoi(line.c_str());
  M = new int*[N];
  for(int i = 0; i < N; i ++){
    M[i] = new int[N];
  }


  int linectr = 0;
  while(getline(input,line)){
    stringstream ss(line);
    int temp;
    int ctr = 0;
    while(ss >> temp)
      M[linectr][ctr++] = temp;

    linectr++;
  }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  unsigned int sharedmem = prop.sharedMemPerBlock;

  cout << "Here are the specs\n";
  cout << "Shared mem per block: " << sharedmem << "\n";
// NEED TO FLATTEN THE ARRAY
double * data_as_array = new double[N*N]();

for(int i=0; i<N; i++){
  for(int j=0; j<N; j++){
    // colum-major order
    data_as_array[i*N + j] = (double)M[j][i];
  }
}

cudaSetDevice(MACHINE_NO);


int size_mat = N*N * sizeof(double);
double *d_mat;
double *d_p;
double *d_XX;
double p = 1.0;
double *x = (double*)malloc(sizeof(double)*N);

for(int i=0;i<N;i++){
  x[i]= M[i][N-1];
  for(int j=0;j<N;j++){
    x[i] -= ((double)M[i][j]/2);
  }
  p *= x[i];
}
unsigned long long int SIZE = (unsigned long long int)1 << (N-1);
unsigned long long int THREADS = 512;
unsigned long long int BLOCKS = 32*1024;



while(SIZE < (THREADS*BLOCKS)){
  if(BLOCKS != 1){
    BLOCKS /=2;
  }else{
    THREADS /=2;
  }
}


unsigned long long int CHUNKY = SIZE/(THREADS*BLOCKS);
cout << "Current thread to block;\n\t THREAD:  " << THREADS
                                << "\n\t BLOCKS:  " << BLOCKS << "\n";
double *x_s = (double*) malloc(sizeof(double)*N);
/*
cout<<"Here is the initial X array: \n";
for(int i=0;i<N;i++){
  printf("%.2lf ",x[i]);
}
printf("\n");
*/
/*
int tid = 2;
long long int sloc = tid*CHUNKY;
int yyy = (sloc >>1)^sloc;

for(int i=0;i<N;i++){
    x_s[i] = x[i];
  for(int k=0;k<N;k++){
    if(((yyy>>k) & 1) == 1) {x_s[i] += M[i][k];}
  }
}

long long int starts = tid*CHUNKY+1;
double local_p = 1.0;

for(int i=starts;i<starts+CHUNKY+1;i++){
  int y = (i>>1) ^ i;
  int yy = ( (i-1)>> 1) ^ (i-1);
  int z = __builtin_ctz(y^yy);
  int s = ((y>>z)& 1) == 1 ? 1:-1;
  int prodsign = (i & 1) == 0 ? 1:-1;
  double dd = 1.0;

  for(int j=0;j<N;j++){
    x_s[j] += (double)(s*M[j][z]);
    dd *= x_s[j];
  }
  local_p += (double)(prodsign*dd);
}
printf("Here is the p in CPU: %.2lf\n",local_p);
*/

/*
cout << "Here is the X array for " << CHUNKY << " \n";
for(int i=0;i<N;i++){
  printf("%.2lf ",x_s[i]);
}
printf("\n");
*/

double *XX = (double* )malloc(sizeof(double)*N*THREADS*BLOCKS);
memset(XX,0.0,sizeof(double)*N*THREADS*BLOCKS);
// memory moving magiac
cout << "Chunky is this: " << CHUNKY << " \n";
cout << "Size is this: " << SIZE << " \n";
cout << "N is this: " << N << " \n";


cudaMalloc((void **)&d_XX,THREADS*BLOCKS*N*sizeof(double));
cudaMalloc((void **)&d_mat,size_mat);
cudaMalloc((void **)&d_p,sizeof(double));
cout << "Memory Allocated...\n";
cudaMemcpy(d_XX,XX,THREADS*BLOCKS*N*sizeof(double),cudaMemcpyHostToDevice);
cudaMemcpy(d_mat,data_as_array,size_mat,cudaMemcpyHostToDevice);
cudaMemcpy(d_p,&p,sizeof(double),cudaMemcpyHostToDevice);
cudaDeviceSynchronize();
cout << "Memory Copied...\n";

// preprocess the fuck out of it
preprocess<<<BLOCKS,THREADS>>>(d_XX,d_mat,SIZE,N,CHUNKY);

cudaMemcpy(XX,d_XX,THREADS*BLOCKS*N*sizeof(double),cudaMemcpyDeviceToHost);
cout << "Preprocess finished running...\n";

double *XXX = (double*) malloc(sizeof(double)*N*THREADS*BLOCKS);
memset(XXX,0.0,sizeof(double)*N*THREADS*BLOCKS);

for(unsigned long long int i=0;i<N;i++){
  for(unsigned long long int j=0;j<THREADS*BLOCKS;j++){
    XXX[j+(i*THREADS*BLOCKS)] = XX[i+(j*N)];
  }
}
cout << "Black magic is finished...\n";
// get it deer
cudaMemcpy(d_XX,XXX,sizeof(double)*N*THREADS*BLOCKS,cudaMemcpyHostToDevice);
/*
cout <<  "Here is the initial XX array: \n";
for(int i=0;i<N;i++){
  printf("%.2lf ",XX[i]);
}
printf("\n");
*/
/*
cout << "Here is the XX array for " << CHUNKY << " \n";
for(int i=0;i<N;i++){
  printf("%.2lf ",XX[N+i]);
}
printf("\n");
*/
cout << "Algo starts now.. Hold on to your seats\n";
double start,end;
start = omp_get_wtime();

 perm_kernel<<<BLOCKS,THREADS>>>(d_XX,CHUNKY,d_p,d_mat,SIZE,N,THREADS,BLOCKS);

cudaMemcpy(&p,d_p,sizeof(double),cudaMemcpyDeviceToHost);
end = omp_get_wtime();
cout << "Kernel finished running...\n";
cout << "Memory re-copied from the device to host...\n";
 p*= (4*(N & 1) - 2);
cout << "Result is: " << p <<" \n";
double result = end-start;
cout <<"The time the kernel took: " << result << " ...\n";


cudaFree(d_XX);
cudaFree(d_mat);
  return 0;
}
