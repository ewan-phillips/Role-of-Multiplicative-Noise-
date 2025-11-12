'''Integrator for stochastic fourth order Runge-Kutta method'''

import numpy as np
from numba import jit 

a21 =   0.66667754298442; a31 =   0.63493935027993; a32 =   0.00342761715422; a41 = - 2.32428921184321
a42 =   2.69723745129487; a43 =   0.29093673271592; a51 =   0.25001351164789; a52 =   0.67428574806272
a53 = - 0.00831795169360; a54 =   0.08401868181222

q1 = 3.99956364361748; q2 = 1.64524970733585; q3 = 1.59330355118722; q4 = 0.26330006501868

@jit(nopython=True)
def srk4_step(x, om, D1, D2, f, g, dt): 
        
        x1, sdt1 = x, np.sqrt(dt*q1)
        k1 = dt * f(x1, om) + sdt1 * g(x1, D1) * np.random.randn() + \
                                    sdt1 * D2 * np.random.randn()
        x2 = x1 + a21 * k1
        
        sdt2 = np.sqrt(dt*q2)
        k2 = dt * f(x2, om) + sdt2 * g(x2, D1) * np.random.randn() + \
                                    sdt2 * D2 * np.random.randn()

        x3 = x1 + a31 * k1 + a32 * k2

        sdt3 = np.sqrt(dt*q3)
        k3 = dt * f(x3, om) + sdt3 * g(x3, D1) * np.random.randn() + \
                                    sdt3 * D2 * np.random.randn()
        
        x4 = x1 + a41 * k1 + a42 * k2 + a43 * k3

        sdt4 = np.sqrt(dt*q4)
        k4 = dt * f(x4, om) + sdt4 * g(x4, D1) * np.random.randn() + \
                                    sdt4 * D2 * np.random.randn()
        
        x = x1 + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4

        return(x)



@jit(nopython=True)
def eul_step(x, om, D1, D2, f, g, dt, sdt):

    x = dt * f(x, om) + sdt * g(x, D1) * np.random.randn() + \
                                    sdt * D2 * np.random.randn()
    
    return(x)



@jit(nopython=True)
def var_step(X, om, D1, D2, f, g, dt, i):

    ss = np.sqrt(dt) * abs(g(X, D1))

    if i%10 == 0:
        if ss > 0.01:
            dt = dt/10 #; print(dt)
        elif ss < 0.0001 and dt < 0.001:
            dt = 10*dt #; print(dt)

    sdt = np.sqrt(dt)

    X += dt * f(X, om) + sdt * g(X, D1) * np.random.randn() + \
                                sdt * D2 * np.random.randn()
    
    # This needs to be fixed for the purpose of producing a histogram.
    # Each value added to the histogram needs to be weighted by the time step somehow. 
    
    return(dt, X)
