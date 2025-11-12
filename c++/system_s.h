#ifndef system_s_h
#define system_s_h

 struct kuramoto_det
 {
     container_type m_omega;
     double m_epsilon;
     int m_range;
     double m_alpha;

     kuramoto_det( const size_t n , double g , double epsilon , double alpha , int range )
     : m_omega( n , 0.0 ) , m_epsilon( epsilon ) , m_alpha( alpha ) , m_range( range )
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

     void set_epsilon( double epsilon ) { m_epsilon = epsilon; }

     double get_epsilon( void ) const { return m_epsilon; }

     void operator()( const container_type &x , container_type &dxdt , double t ) const
     {
         int h = t/0.01;
         
         for( int i=0 ; i<x.size() ; ++i ){

             dxdt[i] = 0;
             
             for( int j=0; j<x.size() ; ++j ){
                 float dist = abs(j-i);
                 dist = abs(dist - round(dist/( (float) x.size() ) ) * ( (float) x.size() ) );
                 if(dist <= m_range && dist > 0){

                     dxdt[i] += ( ( m_epsilon ) /(2*m_range))*sin( x[j] - x[i] - m_alpha );
                 }
             }
         }
     }
 };

 struct kuramoto_stoch
 {
     double m_D;
     int m_range;
     double m_alpha;
     
     kuramoto_stoch( double alpha , int range )
     : m_alpha( alpha ) , m_range( range ) {}
     
     void set_params( double D ) { m_D = D; }
     
     void operator()( const container_type &x , container_type &dxdt , const double t  )
     {
         int h = t/0.01;
         
         for( int i=0 ; i<x.size() ; ++i ){

             dxdt[i] = 0;
             
             for( int j=0; j<x.size() ; ++j ){
                 float dist = abs(j-i);
                 dist = abs(dist - round(dist/( (float) x.size() ) ) * ( (float) x.size() ) );
                 if(dist <= m_range && dist > 0){
                     dxdt[i] += ( m_D /(2*m_range))*sin( x[j] - x[i] - m_alpha );
                 }
             }
         }
     }
 };

/*
 //template<class det, class stoch>
 struct statistics_observer
 {
     double m_K_mean;
     size_t m_count;
     int m_r_one;

     statistics_observer( int r_one = 0 )
     : m_r_one( r_one ) , m_K_mean( 0.0 ) , m_count( 0 ) { }

     template< class State >
     void operator()( const State &x , double t )
     {
         //double r = 0;
         //for(int i=0; i < x.size(); ++i){
         //   r += calc_mean_field_r( x , m_r_one , i ).first;
         //}
         //m_K_mean += r / x.size();

         
         pair< double , double > mean = calc_mean_field( x );
         m_K_mean += mean.first;
         ++m_count;
     }

     double get_K_mean( void ) const { return ( m_count != 0 ) ? m_K_mean / double( m_count ) : 0.0 ; }

     void reset( void ) { m_K_mean = 0.0; m_count = 0; }
 };

/*
 int main( int argc , char **argv )
 {
     const size_t n = 16384;
     //const int r_one = 1000;
     
     //const size_t n = 16384;
     const double dt = 0.002; //0.01;
     
     const double g = atof(argv[1]);

     //container_type x( n );
     container_type x( n );

     boost::mt19937 rng;
     boost::uniform_real<> unif( 0.0 , 2.0 * M_PI );
     boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );

     kuramoto_det det( n , g );
     kuramoto_stoch stoch;
     statistics_observer obs; // ( r_one ) //< kuramoto_det , kuramoto_stoch >
     
     string path = string("/home/iwanphillips/mt4_odeint/results/1layer/1layer_Dfr2_ha1_fnoise_alpha") + string(argv[2]) + ("_g") + string(argv[1]) + string(".txt");
     // chimera_n8192_r1000_1layer_kuramoto_wnoise_g
     ofstream data_out(path);
     cout << path << '\n';

     double alpha_ = atof(argv[2]);
     //double gamma = 1/tau;
     
     //random_device rd;

     for( double D = 0.0 ; D < 2.001 ; D += 0.2 ){
         for( double epsilon = 0.0 ; epsilon < 5.0 ; epsilon += 0.1 ){
             
             int steps = 200.0 / dt;
             int n_pts = 16384;
             while(steps > n_pts){n_pts = 2*n_pts;}
             float ha = 1; // 5
             float Q = ha / (2*pow(6.283185, alpha_));
             float Q_d = Q / pow(dt, 1-alpha_);
             int c = 5;
             int max = c*n_pts;
             while(max < 8000000){max += steps; c += 1;}
             vector<float> X;
             int order = 1;
             for( int i=0 ; i<c ; ++i ){
                 float X_i[order*n_pts];
                 memset(X_i, 0.0, sizeof X_i);
                 long utime; utime=(long)time(NULL); long seed=utime;
                 f_alpha(n_pts, X_i, Q_d, alpha_, &seed);
                 vector<double> Xi (X_i, X_i + sizeof(X_i) / sizeof(int) );
                 X.insert(X.end(), Xi.begin(), Xi.end());
             }
             
             //long utime; utime=(long)time(NULL); long seed=utime;
             //vector<float> X = ou_euler( gamma, 0, gamma, 0.0, dt, 2000000, seed );
             
             det.set_epsilon( epsilon );
             stoch.set_params( D * epsilon );
             obs.reset();

             generate( x.begin() , x.end() , gen );
         
             //for( size_t i = 0 ; i < n ; ++i )
             //{
             //    double pos = i*2*M_PI/(n-1) - M_PI;
             //    double r1 = u_dist(generator);
             //    x[i] = 6*r1*exp(-0.76*pos*pos);
             //}
             
             //stoch_RK srk( 1 );
             stochastic_euler_v se( X );

             integrate_const( se , make_pair( det , stoch ) , x , 0.0 , 100.0 , dt );

             integrate_const( se , make_pair( det , stoch )  , x , 0.0 , 100.0 , dt , boost::ref( obs ) );
             cout     << 0 << "\t" << D << "\t" << epsilon << "\t" << obs.get_K_mean() << endl;
             data_out << 0 << "\t" << D << "\t" << epsilon << "\t" << obs.get_K_mean() << endl;
         }
     }
     
     data_out.close();

     return 0;
 }
 */



#endif /* system_s_h */
