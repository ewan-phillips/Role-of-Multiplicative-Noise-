/* This file can be used to generate power law noise */

#ifndef noise_h
#define noise_h

#include <math.h>
#include <stdio.h>
#include <stddef.h>
#define NR_END 1
#define FREE_ARG char*

void nrerror(char error_text[]) 
{
    fprintf(stderr,"Numerical Recipes run-time error...\n");
    fprintf(stderr,"%s\n",error_text);
    fprintf(stderr,"...now exiting to system...\n");
    exit(1);
}

float *vector_(long nl, long nh)
/* Allocate a float vector with subscript range v[nl..nh] */
{
float *v;
v=(float *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(float)));
if (!v) nrerror((char *) "allocation failure in vector()");
return v-nl+NR_END;
}

void free_vector(float *v, long nl, long nh)
/* Free a float vector allocated with vector() */   // used for memory allocation §1.2
{
free((char*) (v+nl-1)); //free() declared via stdlib.h
}

#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

float ran1(long *idum)
/* Random number generator of Park and Miller. Returns a uniform random deviate between 0.0 and 1.0*/
{
    int j;
    long k;
    static long iy=0;
    static long iv[NTAB];
    float temp;
    
    if (*idum <= 0 || !iy) { // Initialize.
        if (-(*idum) < 1) *idum=1; // Be sure to prevent idum = 0.
        else *idum = -(*idum);
        for (j=NTAB+7;j>=0;j--) { //Load the shue table (after 8 warm-ups).
            k=(*idum)/IQ;
            *idum=IA*(*idum-k*IQ)-IR*k;
            if (*idum < 0) *idum += IM;
            if (j < NTAB) iv[j] = *idum;
        }
        iy=iv[0];
    }
    k=(*idum)/IQ; // Start here when not initializing.
    *idum=IA*(*idum-k*IQ)-IR*k; // Compute idum=(IA*idum) % IM without over
    if (*idum < 0) *idum += IM; // flows by Schrage's method.
    j=iy/NDIV; // Will be in the range 0..NTAB-1.
    iy=iv[j]; // Output previously stored value and refill the shuffle table
    iv[j] = *idum;
    if ((temp=AM*iy) > RNMX) return RNMX; // Because users don't expect endpoint values.
    else return temp;
}

float gas_dev(long *idum)
//Returns a normally distributed deviate with zero mean and unit variance, using ran1(idum)
//as the source of uniform deviates.
{
    float ran1(long *idum);
    static int iset=0;
    static float gset;
    float fac,rsq,v1,v2;
    if (*idum < 0) iset=0; //Reinitialize.
    if (iset == 0) { //We don't have an extra deviate handy, so
        do {
            v1=2.0*ran1(idum)-1.0; //pick two uniform numbers in the square exv2=
            v2=2.0*ran1(idum)-1.0; //tending from -1 to +1 in each direction,
            rsq=v1*v1+v2*v2; //see if they are in the unit circle,
                } while (rsq >= 1.0 || rsq == 0.0); //and if they are not, try again.
                    fac=sqrt(-2.0*log(rsq)/rsq);
//Now make the Box-Muller transformation to get two normal deviates. 
        gset=v1*fac;
        iset=1; //Set flag.
        return v2*fac;
        } else { //We have an extra deviate handy,
            iset=0; //so unset the flag,
        }
    return gset; //and return it.
}


#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr
void four1(float data[], unsigned long nn, int isign)
//Replaces data[1..2*nn] by its discrete Fourier transform
{
    unsigned long n,mmax,m,j,istep,i;
    double wtemp,wr,wpr,wpi,wi,theta;
    float tempr,tempi;
    
    n=nn << 1;
    j=1;
    for (i=1;i<n;i+=2) { //This is the bit-reversal section of the
        if (j > i) {     //routine.
            SWAP(data[j],data[i]); //Exchange the two complex numbers.
            SWAP(data[j+1],data[i+1]);
        }
        m=n >> 1;
        while (m >= 2 && j > m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
    //Here begins the Danielson-Lanczos section of the routine.
    mmax=2;
    while (n > mmax) { //Outer loop executed log2 nn times.
        istep=mmax << 1;
        theta=isign*(6.28318530717959/mmax); // Initialize the trigonometric recurrence.
        wtemp=sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi=sin(theta);
        wr=1.0;
        wi=0.0;
        for (m=1;m<mmax;m+=2) { //Here are the two nested inner loops.
            for (i=m;i<=n;i+=istep) {
                j=i+mmax; //This is the Danielson-Lanczos formula:
                tempr=wr*data[j]-wi*data[j+1];
                tempi=wr*data[j+1]+wi*data[j];
                data[j]=data[i]-tempr;
                data[j+1]=data[i+1]-tempi;
                data[i] += tempr;
                data[i+1] += tempi;
            }
            wr=(wtemp=wr)*wpr-wi*wpi+wr; //Trigonometric recurrence.
            wi=wi*wpr+wtemp*wpi+wi;
        }
        mmax=istep;
    }
}

void realft(float data[], unsigned long n, int isign)
/* Calculates the Fourier transform of a set of n real-valued data points. */
{
    //void four1(float data[], unsigned long nn, int isign); // function declaration is probably not necassary here
    unsigned long i,i1,i2,i3,i4,np3;
    float c1=0.5,c2,h1r,h1i,h2r,h2i;
    double wr,wi,wpr,wpi,wtemp,theta; // Double precision for the trigonometric recurrences.
    
    theta=3.141592653589793/(double) (n>>1); // Initialize the recurrence.
    if (isign == 1) {
        c2 = -0.5;
        four1(data,n>>1,1); // The forward transform is here.
    } else {
        c2=0.5; // Otherwise set up for an inverse transtheta
        theta = -theta; // form.
    }
    wtemp=sin(0.5*theta);
    wpr = -2.0*wtemp*wtemp;
    wpi=sin(theta);
    wr=1.0+wpr;
    wi=wpi;
    np3=n+3;
    for (i=2;i<=(n>>2);i++)
    { // Case i=1 done separately below.
        i4=1+(i3=np3-(i2=1+(i1=i+i-1)));
        h1r=c1*(data[i1]+data[i3]); // The two separate transforms are seph1i=
        h1i=c1*(data[i2]-data[i4]); // arated out of data.
        h2r = -c2*(data[i2]+data[i4]);
        h2i=c2*(data[i1]-data[i3]);
        data[i1]=h1r+wr*h2r-wi*h2i; // Here they are recombined to form
    // the true transform of the original real data.
        data[i2]=h1i+wr*h2i+wi*h2r;
        data[i3]=h1r-wr*h2r+wi*h2i;
        data[i4] = -h1i+wr*h2i+wi*h2r;
        wr=(wtemp=wr)*wpr-wi*wpi+wr; // The recurrence.
        wi=wi*wpr+wtemp*wpi+wi;
    }
    if (isign == 1) {
        data[1] = (h1r=data[1])+data[2]; 
        data[2] = h1r-data[2];
    } else {
        data[1]=c1*((h1r=data[1])+data[2]);
        data[2]=c1*(h1r-data[2]);
        four1(data,n>>1,-1); // This is the inverse transform for the
    } 
}

//----------------------------------------------------------------------------

void f_alpha(int n_pts, float X[], float Q_d, float alpha, long int *idum) 
{
    int i, nn;
    nn = n_pts + n_pts;
    float *hfa, *wfa;
    float ha, wr, wi;
    ha = alpha/2.0;
    Q_d = sqrt( Q_d );
    
    hfa=vector_(1,nn);
    wfa=vector_(1,nn);
    hfa[1] = 1.0;
    wfa[1] = Q_d * gas_dev (idum);
    
    for (i=2; i<=n_pts; i++){
        // generate the coefficients hk
        hfa[i]=hfa[i-1]*(ha + (float) (i-2)) / ((float)(i-1));
        // fill the sequence wk with white noise
        wfa[i] = Q_d*gas_dev(idum);
    }
    // pad the ends of the sequences with zeroes
    for (i=n_pts+1; i <= nn; i++){
        hfa[i] = 0.0;
        wfa[i] = 0.0;
    }
    // perform the discrete Fourier transform
    realft (hfa, n_pts, 1);
    realft (wfa, n_pts, 1);
    // multiply the two complex vectors
    wfa[1] = wfa[1]*hfa[1];
    wfa[2] = wfa[2]*hfa[2];
    for(i=3; i<=nn; i+=2){
        wr = wfa[i];
        wi = wfa[i+1];
        wfa[i] = wr*hfa[i] - wi*hfa[i+1];
        wfa[i+1] = wr*hfa[i+1] + wi*hfa[i];
    }
    // inverse Fourier transform the result
    realft(wfa, n_pts, -1);
    for(i=1; i<=n_pts; i++){
        X[i] += wfa[i]/ ((float) n_pts); 
    }
    free_vector(hfa, 1, nn);
    free_vector(wfa, 1, nn);
}

void cor_exp(int n_pts, float X[], float Q, long int *idum, double dt, double tau)
{
    int i;
    float *xfa, *wfa;
    float Q_d = (Q/(2*tau))*(1 - exp(-2*dt/tau));
    Q_d = sqrt( Q_d );
    
    xfa=vector_(1,n_pts);
    wfa=vector_(1,n_pts);
    xfa[1] = 0.0; 
    wfa[1] = Q_d * gas_dev (idum);
    
    for (i=2; i<=n_pts; i++){
        xfa[i]=xfa[i-1]*exp(-dt/tau) + wfa[i-1];
        wfa[i] = Q_d*gas_dev(idum);
    }

    for(i=1; i<=n_pts; i++){
        X[i] = xfa[i];
    }
    free_vector(xfa, 1, n_pts);
    free_vector(wfa, 1, n_pts);
}

#endif /* noise_h */
