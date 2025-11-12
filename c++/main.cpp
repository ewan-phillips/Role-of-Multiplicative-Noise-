/* This code can be used to generate trajectories for oscillators with smooth colored noise */

#include <iostream>
#include <utility>

#include <boost/numeric/odeint.hpp>
#include <boost/random.hpp>

#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include <cmath>
#include <random>
#include <boost/random.hpp>

#include <string>

using namespace std;
using namespace boost::numeric::odeint;
typedef vector< double > container_type;

#include "noise.h"
#include "ou_noise.h"
#include "stepper_vector.h"
#include "f_stepper.h"
#include "system.h"
#include "observer.h"


int main( int argc , char **argv )
{
    const size_t n = 200;
    const int r_one = 70;
    const double dt = 0.01;
    const double alpha = 1.45;

    container_type x( n );

    boost::mt19937 rng;
    boost::uniform_real<> unif( 0.0 , 2.0 * M_PI );
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );
    
    double tau = 0.02; 
    double epsilon = 1;
    int seed = 377999;
    double gamma = 1/tau;
    double D = 0.003; 
    
    string path = "/Users/tphillips/trial.txt";
    ofstream data_out(path);
    
    int utime; utime=(int)time(NULL); int seed2 = utime;
    vector<float> X = {}; 
        
    phase_ensemble ensemble( X , n , 0 , alpha , r_one );
    observer obs( data_out , epsilon , tau , r_one );
    
    ensemble.set_epsilon( epsilon , D*epsilon );
    obs.set_params( D );
    obs.reset();
    
    default_random_engine generator(seed);
    static uniform_real_distribution<double> u_dist(-0.5,0.5);

    for( size_t i = 0 ; i < n ; ++i )
    {
        double pos = i*2*M_PI/(n-1) - M_PI;
        double r1 = u_dist(generator);
        x[i] = 6*r1*exp(-0.76*pos*pos);
        cout << x[i] << '\n';
    }

    integrate_const( runge_kutta4< container_type >() , boost::ref( ensemble ) , x , 0.0 , 200.0 , dt , boost::ref( obs ) );
    
    data_out.close();

    return 0;
}
