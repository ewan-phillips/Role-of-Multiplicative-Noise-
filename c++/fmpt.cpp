#include <string>
#include <iostream>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>

#include "integrator.h"

double a_, b_;

double pot_sig(double x0) { return((om/sqrt(D1*D2))*atan((sqrt(D1/D2))*x0));}

double psi_m_sig(double x0) { return(pow(D1*x0*x0 + D2, -sig/(2*D1)) * exp(-pot_sig(x0))); }

double psi_d_b_sig(double x0) { return(pow(D1*x0*x0 + D2, (sig/(2*D1)) - 1) * exp(pot_sig(x0))); }

double double_psi_sig(double x0) { return(psi_m_sig(x0)*simp_con(psi_d_b_sig,a_,x0)); }

double double_psi(double x0) { return(psi_m(x0)*simp_con(psi_d_b,a_,x0)); }

double mfpt(double x0) {
    /* T(x): general solution */ 
    double tl = simp_con(psi_m,a_,x0)*simp_con(double_psi,x0,b_);
    double tr = simp_con(psi_m,x0,b_)*simp_con(double_psi,a_,x0);
    return(2*(tl - tr));  
}

double mfpt_ao(double x0) { 
    /* T(x): One absorbing barrier */ 
    return(2*simp_con(double_psi_sig,x0,b_)); }

double mfpt_norm(double f, double norm) { return(f/norm); }

int main() 
{   
    double T, T_n, T_r;

    FILE *d1zeig = fopen("/c++/DATA/anal_FMPT_strat_sig0_a0_bpi2_om1_Dm_Dadd_mon_stp0001.dat","w");

    double Dms [40]; double Das [41]; 
    Das[0] = 0.05;
    for (int i = 0; i < 40; i++){ Das[i+1] = i + 1; }
    for (int i = 1; i < 40; i++){ Dms[i] = i; } 

    om = 1, a_ = 0; b_ = pi2;

    for(double i : Dms){ 
    for(double j : Das){ 
        D1 = i; D2 = j; 
        sig = D1 + 0; // strat.

        double tnorm = simp_con(psi_m,a_,b_); 
        std::cout << D1 << " t norm = " << tnorm << " double psi = " << simp_con(double_psi,0.1,1.5) << '\n'; 
        
        for(double x=-0.01; x<0.01; x+=0.01) 
        { 
            T = 0; // mfpt(x);                                    
            T_n = 0; // mfpt_norm(T , tnorm);  // Full solution (5.5.21 in Gardiner Handbook of Stochastic Methods) 
            T_r = mfpt_ao(x);            // One absorbing barrier (5.5.23)

            printf("x=%f  D1=%f  D2=%f  T=%e  T_n=%e  \n", x, D1, D2, T_r, T_n); 
            fprintf(d1zeig, " %f %f %f %e %e \n", x, D1, D2, T_r, T_n); 
        } 
    } 
    } 
    fclose(d1zeig); 
    return 0; 
} 