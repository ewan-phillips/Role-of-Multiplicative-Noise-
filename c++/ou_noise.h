/* This file can be used to generate Ornstein-Uhlenbeck noise */

#ifndef ou_noise_h
#define ou_noise_h

# include <iomanip>
# include <ctime>
# include <cstring>

double r8_uniform_01 ( int *seed );
double r8_normal_01 ( int *seed );
void timestamp ( );


vector<float> ou_euler ( double theta, double mu, double sigma, double x0, double dt, int n , int seed )
{
    double tmax;
    float *x, *dw;
    int j;

    tmax =  dt * ( double ) ( n );
    
    vector<float> Y(n);

    x=vector_(1,n);
    dw=vector_(1,n);

    x[1] = x0;
    dw[1] = r8_normal_01 ( &seed ) * sqrt ( dt );
    
    for ( j = 2; j <= n; j++ )
    {
        x[j] = x[j-1] + dt * theta * ( mu - x[j-1] ) + sigma * dw[j-1];
        dw[j] = r8_normal_01 ( &seed ) * sqrt ( dt );
    }
    
    for ( j = 1; j <= n; j++ )
    {
        Y[j] = x[j];
    }

    free_vector(x, 1, n);
    free_vector(dw, 1, n);

    return Y;
}



double r8_normal_01 ( int *seed )
{
# define R8_PI 3.141592653589793

  double r1;
  double r2;
  static int seed2 = 0;
  static int seed3 = 0;
  static int used = 0;
  double v1;
  static double v2 = 0.0;

  if ( ( used % 2 ) == 1 )
  {
    if ( *seed != seed2 )
    {
      used = 0;
      seed2 = 0;
      seed3 = 0;
      v2 = 0.0;
    }
  }

  if ( ( used % 2 ) == 0 )
  {
    r1 = r8_uniform_01 ( seed );

    if ( r1 == 0.0 )
    {
      cerr << "\n";
      cerr << "R8_NORMAL_01 - Fatal error!\n";
      cerr << "  R8_UNIFORM_01 returned a value of 0.\n";
      exit ( 1 );
    }

    seed2 = *seed;
    r2 = r8_uniform_01 ( seed );
    seed3 = *seed;
    *seed = seed2;

    v1 = sqrt ( - 2.0 * log ( r1 ) ) * cos ( 2.0 * R8_PI * r2 );
    v2 = sqrt ( - 2.0 * log ( r1 ) ) * sin ( 2.0 * R8_PI * r2 );
  }

  else
  {
    v1 = v2;
    *seed = seed3;
  }

  used = used + 1;

  return v1;
# undef R8_PI
}


double r8_uniform_01 ( int *seed )
{
  int i4_huge = 2147483647;
  int k;
  double r;

  if ( *seed == 0 )
  {
    cerr << "\n";
    cerr << "R8_UNIFORM_01 - Fatal error!\n";
    cerr << "  Input value of SEED = 0.\n";
    exit ( 1 );
  }

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + i4_huge;
  }

  r = ( double ) ( *seed ) * 4.656612875E-10;

  return r;
}


void timestamp ( )
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct std::tm *tm_ptr;
  std::time_t now;

  now = std::time ( NULL );
  tm_ptr = std::localtime ( &now );

  std::strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr );

  std::cout << time_buffer << "\n";

  return;
# undef TIME_SIZE
}


#endif /* ou_noise_h */
