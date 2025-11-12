#ifndef integrator_h
#define integrator_h

#include <stdio.h>
#include <iostream>
#include <complex>
#include <cmath>

using namespace std;

#define pi   3.1415926535897932
#define pi2  1.5707963267948966
#define Phi0 0.0
#define L    6.2831853071795864769 

double om, sig, D1, D2, n;

double simpsons(double f(double x), double a, double b, int n){
  double h,integral,x,sum=0;
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

double simp_con(double f(double x), double a, double b)
{   
    int i = 2;
    double eps = 0.001; 
    double integral, integral_new;

    integral_new = simpsons(f,a,b,i);

    do{
        integral=integral_new;   
        i=i+2;
        integral_new = simpsons(f,a,b,i);
        
    }while(fabs(integral_new-integral)>=eps);

    return integral_new;
}

/* sinusoidal */

double B(double x0) { return(2*(D1*sin(x0)*sin(x0) + D2)); }

double g(double x0) { return(sqrt(2*(D1*sin(x0)*sin(x0) + D2))); }

double f(double x0) { return(om + sig*sin(x0)); }

double pot_;
double sig_;

double psi_p(double x0) 
{ 
  n = floor((x0 + pi2) / pi); 
  pot_ = (om/(sqrt(D2*(D1 + D2)))) * ((n*pi) + atan(sqrt((D1 + D2)/D2)*tan(x0)));

  if(x0 < pi){
    sig_ = pow(tan(x0/2),sig/D1);
  }else{
    sig_ = 1;
  }
  
  return(sig_*exp(pot_)); 
}

double psi_m(double x0) { return(1/psi_p(x0)); }

double psi_d_b(double x0) { return(psi_p(x0)/B(x0)); }

double b_d_psi(double x0) { return(B(x0)/psi_p(x0)); }

double P(double x0) 
{
    return(psi_d_b(x0)*((simp_con(psi_m,Phi0,x0)/psi_d_b(Phi0+L)) + (simp_con(psi_m,x0,Phi0+L)/psi_d_b(Phi0))));   
}

#endif /* integrator_h */