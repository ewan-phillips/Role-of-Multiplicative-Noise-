#ifndef observer_h
#define observer_h

pair< double , double > calc_mean_field( const container_type &x )
{
    size_t n = x.size();
    double cos_sum = 0.0 , sin_sum = 0.0;
    for( size_t i=0 ; i<n ; ++i )
    {
        cos_sum += cos( x[i] );
        sin_sum += sin( x[i] );
    }
    cos_sum /= double( n );
    sin_sum /= double( n );

    double K = sqrt( cos_sum * cos_sum + sin_sum * sin_sum );
    double Theta = atan2( sin_sum , cos_sum );

    return make_pair( K , Theta );
}

pair< double , double > calc_mean_field_r( const container_type &x , int range , int i )
{
    size_t n = x.size();
    double cos_sum = 0.0 , sin_sum = 0.0;
     
    for( int j=0 ; j<n ; ++j){
        float dist = abs(j-i);
        dist = abs(dist - round(dist/( (float) n ) ) * n);
             
        if(dist <= range && dist > 0){
        cos_sum += cos( x[j] );
        sin_sum += sin( x[j] );
        }
    }
    cos_sum /= double( 2 * (float) range );
    sin_sum /= double( 2 * (float) range );
         
     
    double K = sqrt( cos_sum * cos_sum + sin_sum * sin_sum );
    double Theta = atan2( abs(sin_sum) , abs(cos_sum) );
  
    return make_pair( K , Theta );
}

template<class system>
struct observer
{
    ostream&        m_outfile;
    double m_K_mean;
    size_t m_K_count;
    size_t m_count;
    double m_D;
    double m_epsilon;
    double m_tau;
    int m_r_one;
    
    container_type m_dxdt;
    system m_odefun; 
    
    observer( system odefun, container_type &x, ostream &out , double epsilon , double tau , int r_one )
    : m_odefun(odefun) , m_dxdt(x) , m_K_mean( 0.0 ) , m_K_count( 0 ) , m_count( 0 ) , m_outfile( out ) ,
    m_epsilon( epsilon ) , m_tau( tau ) , m_r_one( r_one ) { }
    
    void set_params( double D ) { m_D = D; }
    
    template< class State >
    void operator()( const State &x , double t )
    {
        m_odefun( x, m_dxdt, t );
        
        if(m_count % 100 == 0 ){
            
            double r2 = 0;
            for(int i=0; i < x.size(); ++i){
                r2 += calc_mean_field_r( x , m_r_one , i ).first;
            }
            cout << m_D << '\t' << t << '\t' << r2 / x.size() << '\n';
            m_outfile << m_D << '\t' << t << '\t' << r2 / x.size() << '\n';
            
            /*for(int i=0; i < x.size(); ++i){
                double r = calc_mean_field_r( x , m_r_one , i ).first;
                m_outfile << t << '\t' << i << '\t' << x[i] << '\t' << r << '\n';
            }*/
            
        }
        ++m_count;
    }

    double get_K_mean( void ) const { return ( m_count != 0 ) ? m_K_mean / double( m_count ) : 0.0 ; }

    void reset( void ) { m_K_mean = 0.0; m_count = 0; }
};




template<class det, class stoch>
struct observer_s
{
    ostream&        m_outfile;
    double m_K_mean;
    size_t m_K_count;
    size_t m_count;
    double m_D;
    double m_epsilon;
    double m_tau;
    int m_r_one;
    
    container_type m_dxdt_d;
    container_type m_dxdt_s;
    det m_odedet;
    stoch m_odestoch;
    
    observer_s( det odedet, stoch odestoch, container_type &x, ostream &out , double epsilon , double tau , int r_one )
    : m_odedet(odedet) , m_odestoch(odestoch) , m_dxdt_d(x) , m_dxdt_s(x) , m_K_mean( 0.0 ) , m_K_count( 0 ) , m_count( 0 ) , m_outfile( out ) ,
    m_epsilon( epsilon ) , m_tau( tau ) , m_r_one( r_one ) { }
    
    void set_params( double D ) { m_D = D; }
    
    template< class State >
    void operator()( const State &x , double t )
    {
        m_odedet( x, m_dxdt_d, t );
        m_odestoch( x, m_dxdt_s, t );
        //container_type dxdt = m_dxdt_d + m_dxdt_s;
        
        if(m_count % 100 == 0 ){
            
            double r2 = 0;
            for(int i=0; i < x.size(); ++i){
                r2 += calc_mean_field_r( x , m_r_one , i ).first;
            }
            cout << m_D << '\t' << t << '\t' << r2 / x.size() << '\n';
            m_outfile << m_D << '\t' << t << '\t' << r2 / x.size() << '\n';
            
            /*for(int i=0; i < x.size(); ++i){
                double r = calc_mean_field_r( x , m_r_one , i ).first;
                m_outfile << t << '\t' << i << '\t' << x[i] << '\t' << r <<
                '\t' << m_dxdt_d[i] + m_dxdt_s[i] << '\n';
            }*/
            
        }
        ++m_count;
    }

    double get_K_mean( void ) const { return ( m_count != 0 ) ? m_K_mean / double( m_count ) : 0.0 ; }

    void reset( void ) { m_K_mean = 0.0; m_count = 0; }
};

#endif /* observer_h */
