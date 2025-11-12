#ifndef system_h
#define system_h


struct phase_ensemble
{
    container_type m_omega;
    double m_epsilon;
    vector<float> m_X;
    double m_D;
    double m_Dadd;
    int m_range;
    double m_alpha;

    phase_ensemble( vector<float> X , const size_t n , double g , double alpha , int range = 0 )
    : m_X ( X ) , m_omega( n , 0.0 ) , m_alpha( alpha ) , m_range( range )
    {
        create_frequencies( g );
    }

    void create_frequencies( double g )
    {
        boost::mt19937 rng;
        boost::cauchy_distribution<> cauchy( 0.0 , g );
        boost::variate_generator< boost::mt19937&, boost::cauchy_distribution<> > gen( rng , cauchy );
        generate( m_omega.begin() , m_omega.end() , gen );
    }

    void set_epsilon( double epsilon , double D ) { m_epsilon = epsilon; m_D = D; }

    double get_epsilon( void ) const { return m_epsilon; }

    void operator()( const container_type &x , container_type &dxdt , double t ) const
    {
        int h = t/0.01;
        cout << m_D << '\t' << m_omega[0] << '\n';
        
        for( int i=0 ; i<x.size() ; ++i ){
            //double D_add = m_X[ i % m_X.size() ];
            dxdt[i] = 0; //0.00001*D_add;
            
            for( int j=0; j<x.size() ; ++j ){
                float dist = abs(j-i);
                dist = abs(dist - round(dist/( (float) x.size() ) ) * ( (float) x.size() ) );
                if(dist <= m_range && dist > 0){
                    double D = m_X[ (h + 8000*i + 4000*j) % m_X.size() ];
                    //cout << "t = " << t << " D = " << D << " m_D = " << m_D << '\n'; why is m_D sometimes almost zero?
                    dxdt[i] += ( ( m_epsilon + m_omega[i] + D*m_D ) /(2*m_range))*sin( x[j] - x[i] - m_alpha );
                    //dxdt[i] += ( ( m_epsilon ) /(2*m_range))*sin( x[j] - x[i] - m_alpha );
                }
            }
            
            /*if(t < 50){
                
                for( int j=0; j<x.size() ; ++j ){
                    float dist = abs(j-i);
                    dist = abs(dist - round(dist/( (float) x.size() ) ) * ( (float) x.size() ) );
                    if(dist <= m_range && dist > 0){
                        dxdt[i] += ( m_epsilon/(2*m_range))*sin( x[j] - x[i] - m_alpha );
                    }
                }
                
            } else if (t >= 50 && t < 100) {
            
                for( int j=0; j<x.size() ; ++j ){
                    float dist = abs(j-i);
                    dist = abs(dist - round(dist/( (float) x.size() ) ) * ( (float) x.size() ) );
                    if(dist <= m_range && dist > 0){
                        double D = m_X[ (h + 8000*i) % m_X.size() ];
                        dxdt[i] += ( m_epsilon/(2*m_range))*sin( x[j] - x[i] - m_alpha )  + D*m_D;
                    }
                }
                
            } else {
                
                for( int j=0; j<x.size() ; ++j ){
                    float dist = abs(j-i);
                    dist = abs(dist - round(dist/( (float) x.size() ) ) * ( (float) x.size() ) );
                    if(dist <= m_range && dist > 0){
                        dxdt[i] += ( m_epsilon/(2*m_range))*sin( x[j] - x[i] - m_alpha );
                    }
                }
            }*/
        }
    }
};

#endif /* system_h */
