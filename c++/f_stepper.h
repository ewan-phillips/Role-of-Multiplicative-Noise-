/* Stepper for integrator */

#ifndef f_stepper_h
#define f_stepper_h

#include <boost/random.hpp>
#include <boost/array.hpp>

double r8_normal_01 ( int *seed ); //
double r8_uniform_01 ( int *seed ); //
void timestamp ( ); //

class stochastic_euler
{
    vector<float> m_X;
    
public:

    typedef boost::numeric::ublas::matrix<double> state_type;
    typedef boost::numeric::ublas::matrix<double> deriv_type;
    typedef double value_type;
    typedef double time_type;
    typedef unsigned short order_type;

    typedef boost::numeric::odeint::stepper_tag stepper_category;

    static order_type order( void ) { return 1; }
    
    stochastic_euler( vector<float> X ) : m_X ( X ) {}

    template< class System >
    void do_step( System system , state_type &x , time_type t , time_type dt ) const
    {
        int h = t / dt;
        
        int seed=377999; //
        
        deriv_type det( x.size1() , 2 , 0. ) , stoch( x.size1() , 2 , 0. ) ;
        system.first( x , det , t );
        system.second( x , stoch , t );
        
        for( size_t i=0 ; i<x.size1() ; ++i ){
            for( size_t j=0 ; j<x.size2() ; ++j ){
                //double y = m_X[ (h + i*40000 + j*2000000) % m_X.size() ];
                //x(i,j) += det(i,j) * dt + stoch(i,j) * y * dt;
            }
        }
    }
};


class stochastic_euler_v
{
    vector<float> m_X;
    
public:

    typedef vector< double > state_type;
    typedef vector< double > deriv_type;
    typedef double value_type;
    typedef double time_type;
    typedef unsigned short order_type;

    typedef boost::numeric::odeint::stepper_tag stepper_category;

    static order_type order( void ) { return 1; }
    
    stochastic_euler_v( vector<float> X ) : m_X ( X ) {}

    template< class System >
    void do_step( System system , state_type &x , time_type t , time_type dt ) const
    {
        int h = t / dt;
        
        deriv_type det( x.size() ) , stoch( x.size() ) ;
        system.first( x , det , t );
        system.second( x , stoch , t );
        
        for( size_t i=0 ; i<x.size() ; ++i ){
                double y = m_X[ (h + i*40000) % m_X.size() ];
                x[i] += det[i] * dt + stoch[i] * y * dt;
        }
    }
};


class fbm_milstein
/* Milstein stepper for fractional Brownian motion */
{
    vector<float> m_X;
    
public:

    typedef boost::numeric::ublas::matrix<double> state_type;
    typedef boost::numeric::ublas::matrix<double> deriv_type;
    typedef double value_type;
    typedef double time_type;
    typedef unsigned short order_type;

    typedef boost::numeric::odeint::stepper_tag stepper_category;

    static order_type order( void ) { return 2; }
    
    fbm_milstein( vector<float> X ) : m_X ( X ) {}

    template< class System >
    void do_step( System system , state_type &x , time_type t , time_type dt ) const
    {
        int h = t / dt;
        
        deriv_type det( x.size1() , 2 , 0. ) , stoch( x.size1() , 2 , 0. ) ;
        system.first( x , det , t );
        system.second( x , stoch , t );
        
        deriv_type stoch2( x.size1() , 2 , 0. ) ;
        system.second.order2( x , stoch2 , t );
        
        for( size_t i=0 ; i<x.size1() ; ++i ){
            for( size_t j=0 ; j<x.size2() ; ++j ){
                double y = m_X[ (h + i*700 + j*15000 + 1) % m_X.size() ];
                x(i,j) += det(i,j) * dt + stoch(i,j) * y * dt + 0.5 * stoch2(i,j) * (y * y) * (dt * dt);
            }
        }
    }
};


#endif /* f_stepper_h */
