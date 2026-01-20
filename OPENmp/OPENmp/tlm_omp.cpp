#include<stdlib.h>
#include<stdio.h>
#include<iostream>
#include<fstream>
#include <ctime>
#include<omp.h>
#define M_PI 3.14276
#define c 299792458
#define mu0 M_PI*4e-7
#define eta0 c*mu0
/* BUILD INSTRUCTIONS(For reproducibility) :
    *Option A(Visual Studio) : Enable / openmp in Project Properties->C / C++->Language */
/*In this section the openMp case I use 'pragma omp parallel for'  the reason i use this version rather i use the others is this command makes my code looks very clean and
and also make sure when the loop start again all condition are all initialized.*/
double** declare_array2D(int NX, int NY);

using namespace std;

int main() {
    int NX =1024;
    int NY = 1024;
    int NT = 8192;
    double dl = 1.0;
    double dt = dl / (sqrt(2.0) * c);

    //2D mesh variables
    double I = 0, tempV = 0, E0 = 0, V = 0;
    double** V1 = declare_array2D(NX, NY);
    double** V2 = declare_array2D(NX, NY);
    double** V3 = declare_array2D(NX, NY);
    double** V4 = declare_array2D(NX, NY);

    double Z = eta0 / sqrt(2.0);

    //boundary coefficients
    double rXmin = -1;
    double rXmax = -1;
    double rYmin = -1;
    double rYmax = -1;

    //input / output
    double width = 20.0 * dt * sqrt(2.0);
    double delay = 100.0 * dt * sqrt(2.0);
    int Ein[] = { 10,10 };
    int Eout[] = { 15,15 };

    ofstream output("output_openmp.out");
    double start_time = omp_get_wtime();
    for (int n = 0; n < NT; n++) {

        // source 
        E0 = (1.0 / sqrt(2.0)) * exp(-(n * dt - delay) * (n * dt - delay) / (width * width));
        V1[Ein[0]][Ein[1]] = V1[Ein[0]][Ein[1]] + E0;
        V2[Ein[0]][Ein[1]] = V2[Ein[0]][Ein[1]] - E0;
        V3[Ein[0]][Ein[1]] = V3[Ein[0]][Ein[1]] - E0;
        V4[Ein[0]][Ein[1]] = V4[Ein[0]][Ein[1]] + E0;

        // scatter
#pragma omp parallel for private(I, V)  //the private function can avoid every thread rewrite the same varibles so i create I and V
        for (int x = 0; x < NX; x++) {
            for (int y = 0; y < NY; y++) {
                I = (2.0 * V1[x][y] + 2.0 * V4[x][y] - 2.0 * V2[x][y] - 2.0 * V3[x][y]) / (4.0 * Z);

                V = 2.0 * V1[x][y] - I * Z;         //port1
                V1[x][y] = V - V1[x][y];
                V = 2.0 * V2[x][y] + I * Z;         //port2
                V2[x][y] = V - V2[x][y];
                V = 2.0 * V3[x][y] + I * Z;         //port3
                V3[x][y] = V - V3[x][y];
                V = 2.0 * V4[x][y] - I * Z;         //port4
                V4[x][y] = V - V4[x][y];
            }
        }

        //connect 
#pragma omp parallel for private(tempV)//the private function can avoid every thread rewrite the same varibles so i create tempv
        for (int x = 1; x < NX; x++) {
            for (int y = 0; y < NY; y++) {
                tempV = V2[x][y];
                V2[x][y] = V4[x - 1][y];
                V4[x - 1][y] = tempV;
            }
        }

        //connect
#pragma omp parallel for private(tempV)//the private function can avoid every thread rewrite the same varibles so i create tempv
        for (int x = 0; x < NX; x++) {
            for (int y = 1; y < NY; y++) {
                tempV = V1[x][y];
                V1[x][y] = V3[x][y - 1];
                V3[x][y - 1] = tempV;
            }
        }

        //boundary 
#pragma omp parallel for
        for (int x = 0; x < NX; x++) {
            V3[x][NY - 1] = rYmax * V3[x][NY - 1];
            V1[x][0] = rYmin * V1[x][0];
        }
#pragma omp parallel for
        for (int y = 0; y < NY; y++) {
            V4[NX - 1][y] = rXmax * V4[NX - 1][y];
            V2[0][y] = rXmin * V2[0][y];
        }


        output << n * dt << "  " << V2[Eout[0]][Eout[1]] + V4[Eout[0]][Eout[1]] << endl;   
        if (n % 100 == 0)
           cout << n << endl;

    }
    output.close();
    double end_time = omp_get_wtime();
    cout << "Execution Time: " << end_time - start_time << " seconds" << endl;
    cin.get();




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