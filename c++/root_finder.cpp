/* Calculates roots (maxima and minima) of stationary probability density. This file may be an older version, which needs to be updated */

#include <iostream>
#include <cmath>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <boost/math/tools/roots.hpp>

#include "integrator_comp.h"

namespace tools = boost::math::tools;
using namespace std;

typedef complex<double> dcomp;
dcomp exp_sig_, fin; 
int i;

dcomp psi_p_(double x0) 
{ 
  n = floor((x0 + pi2) / pi); 
  pot_ = (om/(sqrt(D2*(D1 + D2)))) * ((n*pi) + atan(sqrt((D1 + D2)/D2)*tan(x0)));

  n = floor((x0 + pi2) / (2*pi)); 

  i = -1; i = sqrt(i);
  in_1 = - sqrt(D2)*tan(x0/2) - i*sqrt(D1);
  in_2 =   sqrt(D2)*tan(x0/2) - i*sqrt(D1);

  f_1 = atan(in_1/sqrt(D1+D2));
  f_2 = atan(in_2/sqrt(D1+D2));

  sig_ = exp(((2*n*pi) + f_1 + f_2)/(sqrt(D1*(D1 + D2)))); 
  exp_sig_ = sig_*exp(pot_);
  
  return(exp_sig_); 
}

dcomp psi_m_(double x0) { return(1.0/psi_p_(x0)); }

dcomp psi_d_b_(double x0) { return(psi_p_(x0)/B(x0)); }

dcomp b_d_psi_(double x0) { return(B(x0)/psi_p_(x0)); }

double psi_m_r(double x0) { return(1.0/psi_p_(x0).real()); }
double psi_m_i(double x0) { return(1.0/psi_p_(x0).imag()); }
dcomp psi_m_full = simp_con(psi_m_r,Phi0,Phi0+L); 

dcomp num(const double x)
{
  i = -1; i = sqrt(i);
  dcomp p1 = simp_con(psi_m_r,Phi0,x); 
  dcomp p2 = simp_con(psi_m_r,x,Phi0+L); 
  return(p1*b_d_psi_(Phi0+L) + p2*b_d_psi_(Phi0));
}

dcomp num_der(const double x)
{
  return(psi_m_(x)*(b_d_psi_(Phi0+L) - b_d_psi_(Phi0)));
}

dcomp denom(const double x)
{
  return(b_d_psi_(x)*psi_m_full);
}

dcomp denom_der(const double x)
{
  dcomp left = (2*2*D1*sin(x)*cos(x) - 2*f(x))/(psi_p_(x));

  return(left*psi_m_full);
}

double num_d_den(const double x)
{
  dcomp g_ = denom(x).real(); //denom(x);
  fin = (num_der(x)*g_ - num(x)*denom_der(x))/(g_*g_);
  return(fin.real());
}

//

double b, q, pcrit;
 
double root_function(double x) {
  return pow(x,q)/(1+pow(x,q)) - b*x;
}

bool root_termination(double min, double max) {
    return abs(max - min) <= 0.000001;
}

int main()
{
    b = 0.42;
    q = 2.0;

    FILE *d1zeig = fopen("/c++/DATA/trial10.dat","w");

    double Phi, dist;

    for(double Dm=0.2; Dm<7.5; Dm+=0.2)   
    {
        om = 1; D1 = Dm; D2 = 0.0001;

        std::cout << "B/psi(phi0) = " << psi_p_(Phi0) << std::endl;

        std::cout << "B/psi(phi0) = " << b_d_psi_(Phi0) << " B/psi(phi0+L) = " << b_d_psi_(Phi0+L) << std::endl;
        std::cout << "real integral = " << simp_con(psi_m_r,Phi0,Phi0+L) << std::endl;
        std::cout << "imag integral = " << simp_con(psi_m_i,Phi0,Phi0+L) << std::endl;
        std::cout << simp_con(P,Phi0,Phi0+L) << std::endl;

        double xup = (1/2.0)*(om/D1);
        std::cout << "x upp = " << xup << std::endl;

        pair<double, double> result = tools::bisect(num_d_den, 0.0, 1.83, root_termination);
        pcrit = (result.first + result.second)/2;
        cout << "final result = " << pcrit << endl;

        fprintf(d1zeig, " %f %e \n", D1, pcrit);
    }
}