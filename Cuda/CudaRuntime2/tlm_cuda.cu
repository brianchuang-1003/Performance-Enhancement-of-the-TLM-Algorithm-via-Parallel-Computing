#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include<omp.h>

#define M_PI 3.14276
#define c 299792458.0
#define mu0 (M_PI * 4e-7)
#define eta0 (c * mu0)

using namespace std;
/* * Build Instructions:
 * to make sure download cuda version 13 
 */
// --- GPU Kernels ---
//Scatter Kernel  process the energy scattering within each node 
__global__ void scatter_kernel(double* V1, double* V2, double* V3, double* V4, int NX, int NY, double Z) {
    // caculate the 2D coordinate mapping to current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // In this code i use lots of 'if (x < NX && y < NY)' to make sure thread stay in the the mesh
    if (x < NX && y < NY) {
        int idx = x * NY + y; //mapping 2D coordinate to 1D coordinate
        double v1 = V1[idx], v2 = V2[idx], v3 = V3[idx], v4 = V4[idx];
        double I = (2.0 * v1 + 2.0 * v4 - 2.0 * v2 - 2.0 * v3) / (4.0 * Z);
        V1[idx] = (2.0 * v1 - I * Z) - v1;
        V2[idx] = (2.0 * v2 + I * Z) - v2;
        V3[idx] = (2.0 * v3 + I * Z) - v3;
        V4[idx] = (2.0 * v4 - I * Z) - v4;
    }
}
//Connect Kernel : Handles the exchange of field values between adjacent nodes.
__global__ void connect_kernel(double* V1, double* V2, double* V3, double* V4, int NX, int NY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < NX && y < NY) {
        if (x >= 1) {
            int now = x * NY + y;
            int left = (x - 1) * NY + y;
            double temp = V2[now]; V2[now] = V4[left]; V4[left] = temp; //exchange the data through my temp buffer (horizontal)
        }
        if (y >= 1) {
            int now = x * NY + y;
            int down = x * NY + (y - 1);
            double temp = V1[now]; V1[now] = V3[down]; V3[down] = temp;//exchange the data through my temp buffer (vertical) 
        }
    }
}
// Boundary Kernel : Handles reflections at the simulation edges.
__global__ void boundary_kernel(double* V1, double* V2, double* V3, double* V4, int NX, int NY, double rXmin, double rXmax, double rYmin, double rYmax) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                                                                                              
    if (i < NY) { // X boundaries
        V4[(NX - 1) * NY + i] *= rXmax;
        V2[0 * NY + i] *= rXmin;
    }
    if (i < NX) { // Y boundaries
        V3[i * NY + (NY - 1)] *= rYmax;
        V1[i * NY + 0] *= rYmin;
    
}
//Apply_source : to add the pulse to the srcidx   **srcidx =  Ein[0] * NY + Ein[1]; is a 2D mapping to 1D coordinate
__global__ void apply_source_kernel(double* V1, double* V2, double* V3, double* V4, int srcIdx, double E0) {
    V1[srcIdx] += E0;  
    V2[srcIdx] -= E0;  
    V3[srcIdx] -= E0;
    V4[srcIdx] += E0;
}


int main() {
    
    int NX = 1024;
    int NY = 1024;
    int NT = 8192;
    double dl = 1;
    double dt = dl / (sqrt(2.) * c);

    double Z = eta0 / sqrt(2.);

    double rXmin = -1;
    double rXmax = -1;
    double rYmin = -1;
    double rYmax = -1;

    double width = 20 * dt * sqrt(2.);
    double delay = 100 * dt * sqrt(2.);
    int Ein[] = { 10,10 };
    int Eout[] = { 15,15 };
    size_t size = NX * NY * sizeof(double);

    //Host Memory allocation (CPU)
    double* h_V1_out = (double*)malloc(sizeof(double));
    double* h_V2_out = (double*)malloc(sizeof(double));
    double* h_V3_out = (double*)malloc(sizeof(double));
    double* h_V4_out = (double*)malloc(sizeof(double));

    //Device Memory allocation (GPU)
    double* d_V1, * d_V2, * d_V3, * d_V4;
    cudaMalloc(&d_V1,   ); 
    cudaMalloc(&d_V2, size);
    cudaMalloc(&d_V3, size);
    cudaMalloc(&d_V4, size);
    //Reset the GPU memory before start
    cudaMemset(d_V1, 0, size); 
    cudaMemset(d_V2, 0, size);
    cudaMemset(d_V3, 0, size);
    cudaMemset(d_V4, 0, size);

    // to define the amounts of the block,the reason i add 15 is when the NX or NY cant fully devided into 16 the system will truncate and lost few of data so i add
    //15 to add one more block to deal with rest of the data
    dim3 blockSize(16, 16);
    dim3 gridSize((NX + 15) / 16, (NY + 15) / 16);  
    int maxDim = (NX > NY) ? NX : NY;
    int srcIdx = Ein[0] * NY + Ein[1]; //mapping the input 2D to 1D
    int outIdx = Eout[0] * NY + Eout[1];//mapping the output 2D to 1D

    ofstream output("output_cuda.out");

    double start_time = omp_get_wtime();
    for (int n = 0; n < NT; n++) {
        // Source
        double E0 = (1.0 / sqrt(2.0)) * exp(-(n * dt - delay) * (n * dt - delay) / (width * width));
        apply_source_kernel << <1, 1 >> > (d_V1, d_V2, d_V3, d_V4, srcIdx, E0);

        //kernel
        scatter_kernel << <gridSize, blockSize >> > (d_V1, d_V2, d_V3, d_V4, NX, NY, Z);
        connect_kernel << <gridSize, blockSize >> > (d_V1, d_V2, d_V3, d_V4, NX, NY);
        boundary_kernel << <(maxDim + 255) / 256, 256 >> > (d_V1, d_V2, d_V3, d_V4, NX, NY, rXmin, rXmax, rYmin, rYmax);

        //(Memcpy Device to Host)
        cudaMemcpy(h_V2_out, &d_V2[outIdx], sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_V4_out, &d_V4[outIdx], sizeof(double), cudaMemcpyDeviceToHost);

        output << n * dt << "  " << (*h_V2_out + *h_V4_out) << endl;
        if (n % 100 == 0) cout  << n << endl;
    }

    output.close();
    double end_time = omp_get_wtime();
    cout << "Execution Time: " << end_time - start_time << " seconds" << endl;

    cudaFree(d_V1); cudaFree(d_V2); cudaFree(d_V3); cudaFree(d_V4);
    free(h_V1_out); free(h_V2_out); free(h_V3_out); free(h_V4_out);
    return 0;
}