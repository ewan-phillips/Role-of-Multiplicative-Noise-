#ifndef stepper_h
#define stepper_h

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>

#include <vector>
#include <iostream>
#include <boost/random.hpp>
#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>

double r8_normal_01 ( int *seed );
double r8_uniform_01 ( int *seed );
void timestamp ( );


class stoch_RK
{
    double a21 =   0.66667754298442; double a31 =   0.63493935027993; double a32 =   0.00342761715422; double a41 = - 2.32428921184321;
    double a42 =   2.69723745129487; double a43 =   0.29093673271592; double a51 =   0.25001351164789; double a52 =   0.67428574806272;
    double a53 = - 0.00831795169360; double a54 =   0.08401868181222;

    double q1 = 3.99956364361748; double q2 = 1.64524970733585; double q3 = 1.59330355118722; double q4 = 0.26330006501868;
    
    double m_q;
    
public:

    typedef vector< double > state_type;
    typedef vector< double > deriv_type;
    typedef double value_type;
    typedef double time_type;
    typedef unsigned short order_type;

    typedef boost::numeric::odeint::stepper_tag stepper_category;

    static order_type order( void ) { return 4; }
    
    stoch_RK( double q ) : m_q( q ) {}

    template< class System >
    void do_step( System system , state_type &x , time_type t , time_type dt ) const
    {
        int h = t / dt;
        
        double t1 = t;
        state_type x1 = x;
        state_type k1( x.size() );
        deriv_type det1( x.size() ) , stoch1( x.size() )  ;
        system.first( x1 , det1 , t1 );
        system.second( x1 , stoch1 , t1 );
        
        int utime; utime=(int)time(NULL); int seed=utime;
        
        state_type x2( x.size() );
        
        for( size_t i=0 ; i<x.size() ; ++i ){
            double w1 = r8_normal_01 ( &seed ) * sqrt ( q1 * m_q / dt );
            k1[i] = dt * det1[i] + dt * stoch1[i] * w1;
            x2[i] = x1[i] + a21 * k1[i];
        }
        
        double t2 = t1 + a21 * dt;
        state_type k2( x.size() );
        deriv_type det2( x.size() ) , stoch2( x.size()) ;
        system.first( x2 , det2 , t2 );
        system.second( x2 , stoch2 , t2 );
        
        state_type x3( x.size() );
        
        for( size_t i=0 ; i<x.size() ; ++i ){
            double w2 = r8_normal_01 ( &seed ) * sqrt ( q2 * m_q / dt );
            k2[i] = dt * det2[i] + dt * stoch2[i] * w2;
            x3[i] = x1[i] + a31 * k1[i] + a32 * k2[i];
        }
        
        double t3 = t1 + a31 * dt  + a32 * dt;
        state_type k3( x.size() );
        deriv_type det3( x.size() ) , stoch3( x.size() );
        system.first( x3 , det3 , t3 );
        system.second( x3 , stoch3 , t3 );
        
        state_type x4( x.size() );
        
        for( size_t i=0 ; i<x.size() ; ++i ){
                double w3 = r8_normal_01 ( &seed ) * sqrt ( q3 * m_q / dt );
                k3[i] = dt * det3[i] + dt * stoch3[i] * w3;
                x4[i] = x1[i] + a41 * k1[i] + a42 * k2[i] + a43 * k3[i];
        }
        
        double t4 = t1 + a41 * dt  + a42 * dt  + a43 * dt;
        state_type k4( x.size());
        deriv_type det4( x.size() ) , stoch4( x.size() ) ;
        system.first( x4 , det4 , t4 );
        system.second( x4 , stoch4 , t4 );
        
        for( size_t i=0 ; i<x.size() ; ++i ){
            double w4 = r8_normal_01 ( &seed ) * sqrt ( q4 * m_q / dt );
            k4[i] = dt * det4[i] + dt * stoch4[i] * w4;
            
            x[i] = x1[i] + a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i];
        }
    }
};




//// testing below

class stoch_RK_scalar
{
    double a21 =   0.66667754298442; double a31 =   0.63493935027993; double a32 =   0.00342761715422; double a41 = - 2.32428921184321;
    double a42 =   2.69723745129487; double a43 =   0.29093673271592; double a51 =   0.25001351164789; double a52 =   0.67428574806272;
    double a53 = - 0.00831795169360; double a54 =   0.08401868181222;

    double q1 = 3.99956364361748; double q2 = 1.64524970733585; double q3 = 1.59330355118722; double q4 = 0.26330006501868;
    
    double m_q;
    
public:

    double w1, w2, w3, w4;
    double k1, k2, k3, k4;
    double x1, x2, x3, x4;
    double t1, t2, t3, t4;
    double det1, det2, det3, det4;
    double stoch1, stoch2, stoch3, stoch4;
    int h, utime, seed;

    typedef boost::numeric::odeint::stepper_tag stepper_category;

    static unsigned short order( void ) { return 4; }
    
    stoch_RK_scalar( double q ) : m_q( q ) {}

    template< class System >
    void do_step( System system , double &x , double t , double dt ) const
    {
        h = t / dt;
        
        t1 = t;
        x1 = x;
        system.first( x1 , det1 , t1 );
        system.second( x1 , stoch1 , t1 );
        
        utime; utime=(int)time(NULL); seed=utime;
        
        w1 = r8_normal_01 ( &seed ) * sqrt ( q1 * m_q / dt );
        k1 = dt * det1 + dt * stoch1 * w1;
        x2 = x1 + a21 * k1;
        
        t2 = t1 + a21 * dt;
        system.first( x2 , det2 , t2 );
        system.second( x2 , stoch2 , t2 );
        
        w2 = r8_normal_01 ( &seed ) * sqrt ( q2 * m_q / dt );
        k2 = dt * det2 + dt * stoch2 * w2;
        x3 = x1 + a31 * k1 + a32 * k2;
        
        t3 = t1 + a31 * dt  + a32 * dt;
        system.first( x3 , det3 , t3 );
        system.second( x3 , stoch3 , t3 );
        
        w3 = r8_normal_01 ( &seed ) * sqrt ( q3 * m_q / dt );
        k3 = dt * det3 + dt * stoch3 * w3;
        x4 = x1 + a41 * k1 + a42 * k2 + a43 * k3;
        
        t4 = t1 + a41 * dt  + a42 * dt  + a43 * dt;
        system.first( x4 , det4 , t4 );
        system.second( x4 , stoch4 , t4 );
        
        w4 = r8_normal_01 ( &seed ) * sqrt ( q4 * m_q / dt );
        k4 = dt * det4 + dt * stoch4 * w4;    
        x = x1 + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4;
    }
};

#endif /* stepper_h */


