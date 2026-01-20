#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <mpi.h>
#include<omp.h>
#define M_PI 3.14276
#define c 299792458
#define mu0 M_PI*4e-7
#define eta0 c*mu0
/* * =====================================================================
 * BUILD INSTRUCTIONS (For Reproducibility):
 * =====================================================================
 * Option 1 (Visual Studio):
 * The provided Visual Studio solution is pre-configured with MPI paths.
 * Simply Build the 'Release' configuration.
 *
 * EXECUTION:
 * \20737828_Chuang_Pin_Lin_CW2\whole environment to each solution\MPI_test\x64\Release\mpiexec -n 4 MPI_test.exe
 * =====================================================================
 */

double** declare_array2D(int, int);

using namespace std;

int main(int argc, char** argv) {
    
    std::clock_t start = std::clock();
    
    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //create the MPI space
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int NX = 1024;
    int NY = 1024;
    int NT = 8192;
    int local_x = NX / size;                   //size = how many part for calculation
    int X_begin = rank * local_x;              // to define  x in every rank
    int X_end = (rank + 1) * local_x;
    double* temp_buffer = new double[NY];
    double dl = 1;
    double dt = dl / (sqrt(2.) * c);
  

    //2D mesh variables
    double I = 0, tempV = 0, E0 = 0, V = 0;
    double** V1 = declare_array2D(local_x, NY);
    double** V2 = declare_array2D(local_x, NY);
    double** V3 = declare_array2D(local_x, NY);
    double** V4 = declare_array2D(local_x, NY);

    double Z = eta0 / sqrt(2.);

    //boundary coefficients
    double rXmin = -1;
    double rXmax = -1;
    double rYmin = -1;
    double rYmax = -1;

    //input / output
    double width = 20 * dt * sqrt(2.);
    double delay = 100 * dt * sqrt(2.);
    int Ein[] = { 10,10 };
    int Eout[] = { 15,15 };
    
    
    ofstream output;
    if (Eout[0] >= X_begin && Eout[0] < X_end) // to only let the rank which contian the Eout[] to generate the out file
    {
       output.open("output_mpi.out");
    }
    double start_time = omp_get_wtime(); // to make sure the time only start with the loop exclude the memory allowcation
    for (int n = 0; n < NT; n++) {
        if (Ein[0] >= X_begin && Ein[0] < X_end) {

            int local_Ein_x = Ein[0] - X_begin; //to make the coordinate into the relative coordinate system
            int Ein_y = Ein[1];
            //source

            E0 = (1 / sqrt(2.)) * exp(-(n * dt - delay) * (n * dt - delay) / (width * width));
            V1[local_Ein_x][Ein_y] = V1[local_Ein_x][Ein_y] + E0;
            V2[local_Ein_x][Ein_y] = V2[local_Ein_x][Ein_y] - E0;
            V3[local_Ein_x][Ein_y] = V3[local_Ein_x][Ein_y] - E0;
            V4[local_Ein_x][Ein_y] = V4[local_Ein_x][Ein_y] + E0;
        }
        //scatter

        for (int x = 0; x < local_x; x++) {  // the diferent from the serial local_x = the relative x axis coordinate in each ranks 
            for (int y = 0; y < NY; y++) {
                I = (2 * V1[x][y] + 2 * V4[x][y] - 2 * V2[x][y] - 2 * V3[x][y]) / (4 * Z);

                V = 2 * V1[x][y] - I * Z;         //port1
                V1[x][y] = V - V1[x][y];//reflect
                V = 2 * V2[x][y] + I * Z;         //port2
                V2[x][y] = V - V2[x][y];//reflect
                V = 2 * V3[x][y] + I * Z;         //port3
                V3[x][y] = V - V3[x][y];//reflect
                V = 2 * V4[x][y] - I * Z;         //port4
                V4[x][y] = V - V4[x][y]; //reflect
            }
        }

        //connect
        for (int x = 1; x < local_x; x++) //before swap the data in different ranks , i need to deal with the connection inside of ranks first
        {
            for (int y = 0; y < NY; y++) {
                tempV = V2[x][y];
                V2[x][y] = V4[x - 1][y];
                V4[x - 1][y] = tempV;
            }
        }
        for (int x = 0; x < local_x; x++) {
            for (int y = 1; y < NY; y++) {
                tempV = V1[x][y];
                V1[x][y] = V3[x][y - 1];
                V3[x][y - 1] = tempV;
            }
        }
        ///////////////////////////////////////////////////////////////// start MPI PART(outside) ///////////////////////////////////////
        int left = rank - 1; // to make the rank is exist or not 
        int right = rank + 1;//to make the rank is exist or not
        /*I use the temp_buffer to temporary save the data of V2 or V4 when the new value save into V2 or V4 then i will send the buffer data to V4 or V2*/
        MPI_Status status;
        if (left >= 0) {
            MPI_Sendrecv(V2[0], NY, MPI_DOUBLE, left, 0, temp_buffer, NY, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &status); 
            for (int y = 0; y < NY; y++) V2[0][y] = temp_buffer[y];    ///// V2's X and whole vertical data in Y  send first and then receive from the left 
        }
        if (right < size) {
            MPI_Sendrecv(V4[local_x - 1], NY, MPI_DOUBLE, right, 0, temp_buffer, NY, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &status);
            for (int y = 0; y < NY; y++) V4[local_x - 1][y] = temp_buffer[y];/////V4's X and whole vertical data in Y  send first and then receive from the right
        }

        //boundary
        for (int x = 0; x < local_x; x++) {
            V3[x][NY - 1] = rYmax * V3[x][NY - 1];
            V1[x][0] = rYmin * V1[x][0];
        }
        if (rank == 0) {
            for (int y = 0; y < NY; y++) {
                V2[0][y] = rXmin * V2[0][y];   /// rXmin = -1 .reflection
            }
        }
        if (rank == size - 1) {
            for (int y = 0; y < NY; y++) {
                V4[local_x - 1][y] = rXmax * V4[local_x - 1][y];/// rXmax = -1 .reflection
            }
        }


        if (Eout[0] >= X_begin && Eout[0] < X_end) {
            int local_Eout_x = Eout[0] - X_begin;
            output << n * dt << "  " << V2[local_Eout_x][Eout[1]] + V4[local_Eout_x][Eout[1]] << endl;
        }
        if (n % 100 == 0)
            cout << n << endl;

    }
    output.close();
    double end_time = omp_get_wtime();
    cout << "Execution Time: " << end_time - start_time << " seconds" << endl;
    //cin.get();
    if (temp_buffer != NULL) {  // to release the temp buffer at the end of the code
        delete[] temp_buffer;
        temp_buffer = NULL; 
    }
    MPI_Finalize();
    return 0;

}

double** declare_array2D(int NX, int NY) {
    double** V = new double* [NX];
    for (int x = 0; x < NX; x++) {
        V[x] = new double[NY];
    }

    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            V[x][y] = 0;
        }
    }
    return V;
}