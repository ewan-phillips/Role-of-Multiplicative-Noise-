/* ANALYTICAL APPROXIMATION FOR STATIONARY FOKKER PLACK SOLUTION */

//#include <fstream>
//#include <cstdlib>
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
#define L  6.2831853071795864769E0
#define L2 3.1415926535897932
#define N_D 1 
#define NR_END 1
#define FREE_ARG char*
#define EPS 1.0e-4
#define JMAX 100

#define pi 3.1415926535897932
#define pi2 1.5707963267948966
#define pi34 4.7123889803846899
#define del_Phi 0.01 
#define Phi0 -1.5707963267948966 
#define om 1
long double Phi;
long double D1, D2;
long double J;
double n, corr;

double trapzd(double (*func)(double), double a, double b, int n)
{
    /* Trapezoidal method */
    double x,tnm,sum,del;
    static double s;
    int it,j;

    if (n == 1) {
            return (s=0.5*(b-a)*(FUNC(a)+FUNC(b)));
    } else {
            for (it=1,j=1;j<n-1;j++) it <<= 1;
            tnm=it;
            del=(b-a)/tnm;
            x=a+0.5*del;
            for (sum=0.0,j=1;j<=it;j++,x+=del) sum += FUNC(x);
            s=0.5*(s+(b-a)*sum/tnm);
            return s;
    }
}

double qsimp(double (*func)(double), double a, double b)
{
    /* Trapezoidal method with error tolerance */
    double trapzd(double (*func)(double), double a, double b, int n);
    void nrerror(char error_text[]);
    int j;
    double s,st,ost,os;

    ost = os = -1.0e30;
    for (j=1;j<=JMAX;j++) {
            st=trapzd(func,a,b,j);          
            s=(4.0*st-ost)/3.0;
            if (fabs(s-os) < EPS*fabs(os)) return s;
            if (s==0.0 && os==0.0 && j>6) return s;
            os=s;
            ost=st;
    }
    nrerror("Too many steps in routine qsimp");
    return s;
}

void nrerror(char error_text[])
{
        fprintf(stderr,"Numerical Recipes run-time error...\n");
        fprintf(stderr,"%s\n",error_text);
        fprintf(stderr,"...now exiting to system...\n");
        exit(1);
}

long double B(long double x0)
 { 
    /* g^{2}(x) */
    return(2*(D1*sin(x0)*sin(x0) + D2));
 }

long double pot(long double x0) 
 { 
    /* Potential function: int f/g^{2}(x) dx */
    n = floor((x0 + pi2) / pi); 

    return((om/(2*sqrt(D2*(D1 + D2)))) * ((n*pi) + atan(((D1 + D2)/D2)*tan(x0)))); 
 }

long double psi_p(long double x0)  
 { 
    /* psi(phi) = e^(int f/B) */ 
    return(exp(2*pot(x0)));   
 }

long double psi_m(long double x0)  
 { 
    /* psi^(-1)(phi) = e^(-int f/B) */ 
    return(exp(-2.0*pot(x0)));
 }

long double psi_d_b(long double x0) 
 {
    return(psi_p(x0)/B(x0));
 }

long double b_d_psi(long double x0) 
 {
    return(B(x0)/psi_p(x0));
 }

long double simpsons(long double f(long double x), long double a, long double b,int n){
  /* Simpsons method */
  long double h,integral,x,sum=0;
  int i;
  h=fabs(b-a)/n;
  for(i=1;i<n;i++){
    x=a+i*h;
    if(i%2==0){
      sum=sum+2*f(x);
    }
    else{
      sum=sum+4*f(x);
    }
  }
  integral=(h/3)*(f(a)+f(b)+sum);
  return integral;
}

long double simp_con(long double f(long double x), long double a, long double b)
{   
    /* Simpsons method with error tolerance */
    int i = 2;
    long double eps = 0.0001; 
    long double integral, integral_new;

    integral_new = simpsons(f,a,b,i);

    do{
        integral=integral_new;   
        i=i+2;
        integral_new = simpsons(f,a,b,i);
        
    }while(fabs(integral_new-integral)>=eps);

    return integral_new;
}

long double a3_P_(long double x0) 
{
    return(psi_d_b(x0)*((simp_con(psi_m,Phi0,x0)/psi_d_b(Phi0+L)) + (simp_con(psi_m,x0,Phi0+L)/psi_d_b(Phi0))));   
}

long double a3_P_norm(long double x0, long double norm)
{
    return(a3_P_(x0)/norm);
}

double a4_big_int(double x0)
{
    return(psi_p(x0)*simp_con(psi_m,Phi0,Phi0+x0));
}

double a4_P_(double x0) 
{
    double J = (1/2)*(psi_p(Phi0+L) - 1)/a4_big_int(Phi0+L);
    double hom = (2/B(x0))*psi_p(Phi0+L);
    double inhom = - (2/B(x0))*(2*J)*a4_big_int(x0);

    return(hom + inhom);   
}

double a4_P_norm(double x0, double norm)
{
    return(a4_P_(x0)/norm);
}

double a5_P_(double x0) 
{
    double b = om/(sqrt(D2*(D1 + D2)));
    double frac = simp_con(psi_m,Phi0,x0)/simp_con(psi_m,Phi0,Phi0+L);

    return(psi_d_b(x0)*(1 + (1 - psi_p(Phi0+L))*frac));  
}

double a5_P_norm(double x0, double norm)
{
    return(a5_P_(x0)/norm);
}


double a6_P_(double x0) 
{
    double int_pot = simp_con(psi_m,Phi0,x0)/simp_con(psi_m,Phi0,Phi0+L);
    double hom = b_d_psi(Phi0)*(1-int_pot);
    double inhom = b_d_psi(Phi0+L)*int_pot;

    return(psi_d_b(x0)*(hom + inhom));   
}

double a6_P_norm(double x0, double norm)
{
    return(a6_P_(x0)/norm);
}

long double double_psi(long double x0) 
{
    return(psi_m(x0)*simp_con(psi_d_b,0,x0));   
}

long double T_sin(long double x0) 
{
    double tl = simp_con(psi_m,0,x0)*simp_con(double_psi,x0,pi2);
    double tr = simp_con(psi_m,x0,pi2)*simp_con(double_psi,0,x0);
    return(2*(tl - tr));  
}

long double T_sin_norm(long double x0, long double norm)
{
    return(T_sin(x0)/norm);
}

//

int main() 
{   
    /* Probability density */
    long double I, PdB, dist, n_dist;

    FILE *d1zeig = fopen("/c++/DATA/stat_nFPE_om1_Dm2_Dadd_mpi_a3_2.dat","w");

    long double Dms [4] = { 1, 2, 5, 10 }; 
    long double Das [4] = { 0.2, 0.5, 1, 2 };

    for(long double i : Dms){ 
    for(long double j : Das){
        D1 = i; D2 = j; 
       
        Phi=Phi0; 
        long double pnorm = simp_con(a3_P_,Phi0,Phi0+L);
        //double pnorm = qsimp(a2_P_,Phi0,Phi0+L);
        //double J = (1/2)*(psi_p(L) - 1)/a_big_int(L);

        //long double tnorm = simp_con(psi_m,0.1,1.5); //simp_con(psi_m,-pi2,pi2);
        //std::cout << D1 << " t norm = " << tnorm << " double psi = " << simp_con(double_psi,0.1,1.5) << '\n'; 

        for(int i2=0;Phi<Phi0+L2;i2=i2+1) 
        {
            dist = 0; // P(Phi); //dist = T_sin_norm(pi2, tnorm);
            n_dist = a3_P_norm(Phi, pnorm);  //n_dist = T_sin_norm(pi2, tnorm); //n_dist = T_sin_norm(-pi2, tnorm);

            I = pot(Phi);  
            PdB = psi_d_b(Phi);  

            printf("Phi=%f  D1=%f  D2=%f  dist=%e  n_dist=%e  \n", Phi, D1, D2, dist, n_dist);
            fprintf(d1zeig, " %f %f %f %e %e %e %e \n", Phi, D1, D2, dist, n_dist, I, PdB);

            Phi=Phi+del_Phi;
        } 
    }
    }
    fclose(d1zeig);
    return 0;
}