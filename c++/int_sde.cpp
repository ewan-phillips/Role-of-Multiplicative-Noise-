/* Integrate stochastic differential equation (SDE) */

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

#define _2pi 6.2831853071795864769
#define pi 3.1415926535897932

#include "stepper_vector.h"

template<class det, class stoch>
struct observer_s
{
    vector< double > P; 
    ostream&        m_outfile;
    double m_st; double m_norm; double m_xm;
    int k;
    
    observer_s( double &x, ostream &out, const size_t N, double st, double norm, double xm )
    : m_count( 0 ) , m_P( N , 0.0 ), m_outfile( out ), m_st( st ), m_norm( norm ), m_xm( xm )  { }
    
    void set_params( double D ) { m_D = D; }
    
    template< class State >
    void operator()( const State &x , double t )
    {   
        if(m_count > 20000 ){

                X = X%(_2pi); 
                if(X >= pi){ X -= _2pi; } 

                k = floor((X-xm)/del_x);
                m_P[k] += norm; 
        }
        ++m_count;
    }

    double get_P( void ) const { return m_P ; }

    void reset( void ) { m_P = 0.0; m_count = 0; }
};

 struct kuramoto_det
 {
     double m_omega;

     kuramoto_det( double omega ) : m_omega( omega ) {}

     void operator()( const double &x , double &dxdt , double t ) const
     {
        dxdt = m_omega; 
     }
 };

 struct kuramoto_stoch
 {
     double m_D1, m_D2;
     
     //kuramoto_stoch( ) : {}
     void set_params( double D1, double D2 ) { m_D1 = D1; m_D2 = D2; }
     
     void operator()( const double &x , double &dxdt , const double t  )
     {
         dxdt = m_D1 * sin( x ) + m_D2;
     }
 };


int main( int argc , char **argv )
{
    const double T = 100.0;
    const double dt = 0.01;
    const int n = (int) T / dt; 
    const double om = 1;
    double D1, D2;

    const int N = 300;
    double del_x = _2pi/N;
    int st = 20000;
    double norm = 1/(del_x*(n-st));

    double x = 1;
    double xm = -pi; double xp = pi;

    boost::mt19937 rng;
    boost::uniform_real<> unif( 0.0 , 2.0 * M_PI );
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );
    
    string path = "/Users/tphillips/trial.txt";
    ofstream data_out(path);
    
    kuramoto_det det( om );
    kuramoto_stoch stoch;
                
    observer_s< kuramoto_det , kuramoto_stoch > obs( x, data_out, N, st, norm, xm );

    stoch_RK_scalar srk( 1 );
    //stochastic_euler se( X );
    //fbm_milstein mil( X );         

    double Dms [8] = { 0.1, 1, 2, 5, 10, 30, 50, 100 };
    double Das [1] = { 0.8 };

    for(double i : Dms){ 
        for(double j : Das){

            D1 = i; D2 = j; 
                
            stoch.set_params(D1, D2);
            obs.set_params(D1, D2); 
            obs.reset();

            integrate_const( srk, make_pair( det , stoch ), x, 0.0, sim_time, dt, boost::ref( obs ));

        }
    }
    data_out.close();

    return 0;
}





                




