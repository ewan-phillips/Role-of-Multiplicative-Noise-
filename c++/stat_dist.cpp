/* This file can be used to obtain semi analytically the stationary probability distribution from the Fokker-Planck equation */

#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <utility>

#include <fstream>
#include <cstdlib>
#include <stdlib.h>

#include <string>
#include <iostream>
#include<stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include<math.h>
#include<errno.h>
#define FUNC(x) ((*func)(x))
#include<stdlib.h>
#include<time.h>

#include "integrator_comp.h"

#define N_D 1 
#define NR_END 1
#define FREE_ARG char*
#define EPS 1.0e-4
#define JMAX 100

#define pi34 4.7123889803846899
#define del_Phi 0.001 
double Phi;
double a_;
double J;

using namespace std;

double P_(double x0) 
{
    double po = psi_p(x0).real();
    return(po/B(x0));
}

double P_norm(double x0, double norm)
{
    return(P_(x0)/norm);
}

double P_per(double x0) 
{
    return(P(x0).real());  
}

double P_per_norm(double x0, double norm)
{
    return(P_per(x0)/norm);
}

int main() 
{   
    double I, dist, n_dist, per_dist;

    sig = -0.2; 
    om = 1;
    i_ = -1; i_ = sqrt(i_);

    FILE *d2zeig = fopen("/c++/DATA/stat_nFPE_lin_om1_sigm0p2_Dm_Dadd0p001_comp.dat","w");

    double Dms [3] = { 5 }; //{ 0.1, 1, 2, 5, 10, 30, 50, 100 };
    double Das [1] = { 0.001 }; // 0.1

    for(double i : Dms){ 
    for(double j : Das){
        D1 = i; D2 = j;

        Phi=Phi0;

        for(int i2=0;Phi<Phi0+L;i2=i2+1) 
        {
            dist = 0; // P_(Phi); 
            n_dist = 0; // P_norm(Phi, pnorm);
            I = 0; //pot(Phi);   

            per_dist = P_per(Phi); 

            if (i2 % 100 == 0){ 
                printf("Phi=%f pot=%e tan=%e per_dist=%e atan=%e pow=%e tan=%e \n", Phi, psi_p(Phi).real(), tan(Phi), per_dist, atan(sqrt((D1 + D2)/D2)*tan(Phi)), pow(abs(tan(Phi/2)),sig/D1), tan(Phi/2));
            } 
            fprintf(d2zeig, " %f %f %f %e %e %e %e \n", Phi, D1, D2, dist, n_dist, I, per_dist); 
            
            Phi=Phi+del_Phi;
        } 
    }
    }
    fclose(d2zeig);

    return 0;
}