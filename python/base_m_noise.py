import importlib

import numpy as np
from math import pi
import math 
import matplotlib.pyplot as plt
import pandas as pd
import random

from scipy.integrate import solve_ivp
from scipy.signal import welch
import time
import json
import sys
import networkx as nx   
from numba import jit 

import scipy
from scipy.signal import find_peaks
from scipy import signal
from scipy.stats import gaussian_kde
from scipy import special as s

from statsmodels.tsa.stattools import acf

from matplotlib.colors import Normalize
import matplotlib.cm as cm
#from matplotlib import rc

import seaborn as sns
import collections.abc as ca
import sympy as sp

from datetime import datetime
from collections import deque

from IPython.display import clear_output

from stoch_rk import *
import stoch_rk; importlib.reload(stoch_rk)

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

params = {'xtick.labelsize': 22, 'ytick.labelsize': 22,
              'legend.fontsize': 20, 'axes.labelsize': 22,
              'axes.titlesize': 22,  'font.size': 22, 
              'legend.handlelength': 2}

def osc_mod(x):
    """Mod function for oscillators with period 2*pi"""
    phi = [s%(2*pi) for s in x]

    for k in range(len(phi)):
        if(phi[k] >= pi):
            phi[k] = phi[k] -2*pi

    return(phi)

def rotate(l, n):
    return l[n:] + l[:n]

@jit(nopython=True)
def f_(X, om, lam=0):
    """Constant drift function f(x)=om"""
    return(om)

@jit(nopython=True)
def f_sin(X, om, lam=0):
    """Drift function with sinusoidal component"""
    return(om + lam*np.sin(X))

@jit(nopython=True)
def f_lin(X, om, lam=0):
    """Drift function with linear term"""
    return(om + lam*X)

@jit(nopython=True)
def f_mon(X, om, n=0): 
    """Drift function with monomial term"""
    return(om * (abs(X)**n))

@jit(nopython=True)
def f_mon_gen(X, a, n=[0,1]): 
    """Drift function with two monomial terms"""
    return((a[0]*(abs(X)**n[0])) + (a[1]*(abs(X)**n[0])))

@jit(nopython=True)
def f_mon_(X, om, n=0): 
    return(np.sign(X) * om * (abs(X)**n))

@jit(nopython=True)
def g_mon(X, D, m=1):
    """State dependent diffusion function"""
    return(D * (abs(X)**m))

@jit(nopython=True)
def g_mon_der(X, D, m=1):
    """Derivative of g_mon(x)"""
    return(D * m * (abs(X)**(m-1)))

@jit(nopython=True)
def g_mon_fix1(X, D, m=1):
    return(D * (abs(X-1)**m))

@jit(nopython=True)
def g(X, D, m=None):
    """Sinusoidal diffusion function"""
    return(D * np.sin(X))

@jit(nopython=True)
def g_lin(X, D, m=None):
    return(D * X)

@jit(nopython=True)
def g_lin_p(X, D, m=None):

    if X >= np.pi/2: X -= np.pi
    elif X <= -np.pi/2: X += np.pi
    return(D * X)

def labels(ax, osc=True, ylab='theta', ylim=None):
    """Labels plot axes. Osc=True is used for phase oscillators"""

    if osc==True:
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels([r'$-\pi$', r'0', r'$\pi$'])
        ax.set_xlabel(r'$\theta$')
        if ylab=='theta':
            ax.set_ylabel(r'$P(\theta)$')
        elif ylab=='theta_x':
            ax.set_ylabel(r'$P(\theta), P(x)$')

    else:
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$P(x)$')

    if ylim: ax.set_ylim([ylim[0],ylim[1]])

def tlabels(ax):
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
    ax.set_ylabel(r'$\phi$')
    ax.set_xlabel(r'$t$')

def find_max(x, y):
    index_max = np.argmax(y)
    return(x[index_max])

def FPE_ev(om, D1, D2, T, f, g, dt=0.001, 
           xinit=10, ntrials=400000, s=False, n=0, m=1):
    """Show evolution of probability density"""

    n_ = int(T / dt)  
    D1_, D2_ = np.sqrt(2*D1), np.sqrt(2*D2)
    sdt = np.sqrt(dt)

    ntrials = ntrials
    X = xinit*np.array(np.ones(ntrials))

    snap = [int(0.005*n_), int(0.05*n_), int(0.95*n_)]
    bins = np.arange(0., 16., 16/200) 

    if s: fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    tl, meanl = [], []

    peak, fwhm = [], []

    for i in range(n_):
        X += dt * f(X, om) + sdt * g(X, D1_) * np.random.randn(ntrials) + \
                                   sdt * D2_ * np.random.randn(ntrials)
        
        if i in snap: 
            hist, _ = np.histogram(X, bins=bins)

            peak.append(max(hist))
            lower_x, upper_x = half_max_x((bins[1:] + bins[:-1]) / 2, hist)
            fwhm.append(upper_x - lower_x)

            if s:
                ax.plot((bins[1:] + bins[:-1]) / 2, hist, label=f"t={i * dt:.2f}")
                ax.axhline(max(hist), color='r')
            
        if i%10==0: 
            meanl.append(np.mean(X))
            tl.append(round(i*dt, 4))

            b = np.count_nonzero(X < 0)
            if b > 0: print("(Sims < 0) = ", (b/ntrials)*100, "%")

    clear_output(wait=True)

    if s:
        fig.show() 
        ax.legend() 
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$p(x)$')

    return(tl, meanl, peak, fwhm)

@jit(nopython=True) 
def SDE_ev1(om, D1, D2, T, f, g, dt=0.001, xinit=10, td=True, steps=1, osc=False):
    """Show evolution of SDE considering Itô interpretation"""

    n = int(T / dt)  
    D1_, D2_ = np.sqrt(2*D1), np.sqrt(2*D2)
    sdt = np.sqrt(dt)

    tl, Xl = [], []
    X = xinit

    step = int(T/(dt*10000*steps))
    if step == 0: step +=1

    if td==True:
        for i in range(n):
            dx = dt * f(X, om) + sdt * g(X, D1_) * np.random.randn() + \
                                   sdt * D2_ * np.random.randn()
            if math.isnan(dx): continue 
            X += dx

            if i%step==0: 
                if osc:
                    X = (X + 2*pi)%(2*pi) 
                    if(X >= pi): X -= 2*pi

                Xl.append(X)
                tl.append(i*dt)

        return(tl, Xl)
    else:    
        for i in range(n):
            X += dt * f(X, om) + sdt * g(X, D1_) * np.random.randn() + \
                                        D2_ * np.random.randn()
            if i%step==0: 
                if osc:
                    X = (X + 2*pi)%(2*pi) 
                    if(X >= pi): X -= 2*pi

                Xl.append(X)
                tl.append(i*dt)

        return(tl, Xl)

@jit(nopython=True) 
def SDE_ev(om, D1, D2, T, f, g, dt=0.001, xinit=10, 
           n=0, m=1, osc=False, steps=1, meth='ito', g_der=None):
    """Show evolution of SDE"""

    n_ = int(T / dt)  
    D1_, D2_ = np.sqrt(2*D1), np.sqrt(2*D2)
    sdt = np.sqrt(dt)

    X = xinit
    tl, Xl = [], []

    step = int(T/(dt*10000*steps))

    if meth=='ito':
        print("method = Ito")
        for i in range(n_):

            dx = dt * f(X, om, lam=n) + sdt * g(X, D1_) * np.random.randn() + sdt * D2_ * np.random.randn() 
 
            if math.isnan(dx): continue 
            X += dx 

            if i%step==0: 
                if osc:
                    X = (X + 2*pi)%(2*pi) 
                    if(X >= pi): X -= 2*pi

                Xl.append(X)
                tl.append(i*dt)

    if meth=='strat':
        print("method = Strat.")
        for i in range(n_):
            dx = dt * (f(X, om, lam=n) + 0.5*g_der(X, D1_, m)*g(X, D1_, m)) + \
                 sdt * g(X, D1_, m) * np.random.randn() + \
                                    sdt * D2_ * np.random.randn()

            if math.isnan(dx): continue
            X += dx

            if i%step==0: 
                if osc:
                    X = (X + 2*pi)%(2*pi) 
                    if(X >= pi): X -= 2*pi

                Xl.append(X)
                tl.append(i*dt)

    return(tl, Xl)

@jit(nopython=True) 
def SDE_ev_strat(om, D1, D2, T, f, g, dt=0.001, xinit=10, n=0, m=1, osc=False, steps=1):
    """Show evolution of SDE considering Stratonovich interpretation"""

    n_ = int(T / dt)  
    D1_, D2_ = np.sqrt(2*D1), np.sqrt(2*D2)
    sdt = np.sqrt(dt)

    X = xinit
    tl, Xl = [], []

    step = int(T/(dt*10000*steps))
    print(step)

    for i in range(n_):        
        dx = dt * f(X, om, n) + sdt * g(X, D1_, m) * np.random.randn() + \
                                    sdt * D2_ * np.random.randn()

        if math.isnan(dx): continue
        X += dx

        if i%step==0: 
            if osc:
                X = (X + 2*pi)%(2*pi) 
                if(X >= pi): X -= 2*pi

            Xl.append(X)
            tl.append(i*dt)

    return(tl, Xl)

@jit(nopython=True)
def stat_den_sde(om, D1, D2, T, f_, g, dt, osc=False, Nb=300, low=0, lim=np.pi, xinit=0.1, n=0, m=1, start=50000, sig=None): 
    """Obtain stationary probability distribution"""

    if sig != 0: n = sig

    n_ = int(T / dt)  
    st = start
    Pl = np.zeros(Nb) 
    D1_, D2_ = np.sqrt(2*D1), np.sqrt(2*D2)
    sdt = np.sqrt(dt)
    X = xinit 

    # If periodic boundaries
    if osc:
        xm = -lim       
        del_x = 2*lim/Nb
        x = np.arange(-lim,lim,del_x)
        X = 2*lim*np.random.rand() - lim
        norm = 1/(del_x*(n_-st))

        for i in range(n_):
            X += dt * f_(X, om, n) + sdt * g(X, D1_, m) * np.random.randn() + \
                                    sdt * D2_ * np.random.randn()

            X = (X + 2*lim)%(2*lim) 
            if(X >= lim): X -= 2*lim 

            if math.isnan(X): X = 2*lim*np.random.rand() - lim

            if i > st: 
                k = int(np.floor((X-xm)/del_x))
                Pl[k] += norm 
    else:  # Assuming natural boundaries
        b = lim 
        del_x = abs(b-low)/Nb     
        x = np.arange(low,b,del_x)
        X = xinit #
        norm = 1/(del_x*(n_-st))

        for i in range(n_):
            X += dt * f_(X, om, n) + sdt * g(X, D1_, m) * np.random.randn() + \
                                    sdt * D2_ * np.random.randn()

            if i > st:  
                k = int(np.floor((X)/del_x)) 
                if (abs(k) < Nb):
                    Pl[k + int(abs(low)/del_x)] += norm   

    return(x, Pl)

@jit(nopython=True)
def stat_den_sde_gen(a1, a2, D1, D2, T, f_, g, dt, osc=False, Nb=300, low=0, lim=np.pi, xinit=0.1, n=0, m=1, start=50000): 
    """Obtain stationary probability distribution. Rough and ready way 
       of working with drift functions containing higher order terms"""

    n_ = int(T / dt)  
    st = start
    Pl = np.zeros(Nb) 
    D1_, D2_ = np.sqrt(2*D1), np.sqrt(2*D2)
    sdt = np.sqrt(dt)
    X = xinit 
    m_ = (2*m)-1

    if osc:
        xm = -lim       
        del_x = 2*lim/Nb
        x = np.arange(-lim,lim,del_x)
        X = 2*lim*np.random.rand() - lim
        norm = 1/(del_x*(n_-st))

        for i in range(n_):

            X += dt * (f_(X, a1, n) + f_(X, a2, m_)) + \
                 sdt * g(X, D1_, m) * np.random.randn() + sdt * D2_ * np.random.randn()

            X = (X + 2*lim)%(2*lim) 
            if(X >= lim): X -= 2*lim 

            if math.isnan(X): X = 2*lim*np.random.rand() - lim

            if i > st: 
                k = int(np.floor((X-xm)/del_x))
                Pl[k] += norm 
    else:   
        b = lim 
        del_x = abs(b-low)/Nb     
        x = np.arange(low,b,del_x)
        X = xinit 
        norm = 1/(del_x*(n_-st))

        for i in range(n_):
            dx = dt * (f_(X, a1, n) + f_(X, a2, m_)) + \
                 sdt * g(X, D1_, m) * np.random.randn() + sdt * D2_ * np.random.randn()
            
            if math.isnan(dx): continue
            X += dx
            if X < 0: X = abs(X) 

            if i > st:  
                k = int(np.floor((X)/del_x))

                if (abs(k) < Nb):
                    Pl[k + int(abs(low)/del_x)] += norm   

    return(x, Pl)

def num_fpe_t(om, D1, D2, T, f_, g, dt=0.01, osc=True, ylim=None, save=None):
    """Compare stationary prob. densities generated from SDE's"""

    f, ax = plt.subplots(1, 1, figsize=(8, 4))

    if isinstance(D1, ca.Sequence):
        for i in range(len(D1)):
            x, Pl = stat_den_sde(om, D1[i], D2, T, f_, g, dt, osc)
            ax.plot(x, Pl, label=r"$D_1=$"+str(D1[i]))     

    elif isinstance(D2, ca.Sequence):
        for i in range(len(D2)):
            x, Pl = stat_den_sde(om, D1, D2[i], T, f_, g, dt, osc)
            ax.plot(x, Pl, label=r"$D_2=$"+str(D2[i]))
    else:
        x, Pl = stat_den_sde(om, D1, D2, T, f_, g, dt, osc)
        ax.plot(x, Pl, label=r"$D_1=$"+str(D1)+r"$, D_2=$"+str(D2))

    labels(ax, osc, ylim)
    ax.legend(fontsize=17)

    if save: 
        f.savefig('/'+save+'.png', bbox_inches='tight', format='png', dpi=120)


def anal_fpe(Dm_list, file_, ylim=None, save=None):
    """Compare stationary prob. densities from stationary Fokker Planck equation"""
    
    f, ax = plt.subplots(1, 1, figsize=(8, 4))

    for i in range(len(Dm_list)):
        df=pd.read_csv("/c++/DATA/"+file_+".dat",sep=" ",header=None)
        df = df.dropna(axis='columns',how='all')
        df.columns=['phi','D1','D2','P', 'Pn', 'I', 'PdB']
        df = df[df['D1'] == Dm_list[i]]

        y = np.array(df['Pn'])
        x = np.arange(-np.pi, np.pi, 2*np.pi/len(y))
        ax.plot(x, y, '-', alpha=0.6, label=r'$D_1 = {}$'.format(Dm_list[i]))

    labels(ax, True, ylim)
    ax.legend(fontsize=17)

    if save: 
        f.savefig('/anal_FPE_om1_Dm1_Daddvaried.png', bbox_inches='tight', format='png', dpi=120)


def num_an_fpe(om, D1, D2, T, f_, g, file_, D1a=None, D2a=None, dt=0.01, osc=True, ylim=None, save=None, Nb=300):
    """Compare stationary prob. densities of SDE's (numerical) and stat. FPE (analytical)"""

    sns.set_theme(style="white", rc=params)
    f, ax = plt.subplots(1, 1, figsize=(8, 4))

    if isinstance(D1, ca.Sequence):
        for i in range(len(D1)):
            x, Pl = stat_den_sde(om, D1[i], D2, T, f_, g, dt, osc, Nb=Nb)
            ax.plot(x, Pl, '.', markersize=1, label=r"Num. $D_1=$"+str(D1[i]))    

    elif isinstance(D2, ca.Sequence):
        for i in range(len(D2)):
            x, Pl = stat_den_sde(om, D1, D2[i], T, f_, g, dt, osc, Nb=Nb)
            ax.plot(x, Pl, label=r"Num. $D_2=$"+str(D2[i]))
    else:
        x, Pl = stat_den_sde(om, D1, D2, T, f_, g, dt, osc, Nb=Nb)
        ax.plot(x, Pl, label=r"Num. $D_1=$"+str(D1)+r"$, D_2=$"+str(D2))

    for i in range(len(D1a)):
        df=pd.read_csv("/c++/DATA/"+file_+".dat",sep=" ",header=None)
        df = df.dropna(axis='columns',how='all')
        df.columns=['phi','D1','D2','P', 'Pn', 'I', 'PdB']
        if D1a: df = df[df['D1'] == D1a[i]]
        if D2a: df = df[df['D2'] == D2a]

        y = np.array(df['Pn'])
        x = np.arange(-np.pi, np.pi, 2*np.pi/len(y))
        ax.plot(x, y, '.', markersize=1, alpha=0.6, label=r'Anal. $D_1 = {}$'.format(D1a[i]))

    labels(ax, osc, ylim)
    ax.legend(fontsize=17)

    if save: 
        f.savefig('/'+save+'.png', bbox_inches='tight', format='png', dpi=120)

### FWHM ###

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]

    if len(zero_crossings_i) == 0:
        return [x[np.argmax(y)], x[np.argmax(y)]]
    if len(zero_crossings_i) == 1:
        return [lin_interp(x, y, zero_crossings_i[0], half),
                np.max(x)]
    
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

### FMPT ###

@jit(nopython=True)
def burst_counter(om, D1, D2, T, f, g, dt, a2=0, n_=0, m_=1, X0=0, a=-np.pi/2,
                  b=np.pi/2, act='ref', steps=1): 
    """Count the number of 'bursts' or trajectories that reach a given threshold using resetting"""

    n = int(T / dt)
    D1_, D2_ = np.sqrt(2*D1), np.sqrt(2*D2) 
    X, sdt = X0, np.sqrt(dt) 

    tl, Xl = [], []
    bursts = np.zeros(4) 
    m_2 = (2*m_)-1
    c1  = a2 + (D1*m_)

    step = int(T/(dt*10000*steps))
    if step == 0: step +=1

    for i in range(n): 
        dx = dt * (f(X, om, n_) + f(X, c1, m_2)) + \
                sdt * g(X, D1_, m_) * np.random.randn() + sdt * D2_ * np.random.randn()

        if math.isnan(dx): print("nan at ", i); continue 
        X += dx

        if (X <= a): 
            if act=='ref': 
                #X = a + abs(a-X)
                X = a + abs(abs(a) - abs(X)) 
            elif act=='res':    
                X = X0 
                bursts[1] += 1
            
        if (X >= b):
                bursts[0] += 1
                X = X0 

        if i%step==0: 
            Xl.append(X)
            tl.append(i*dt)

    return(tl, Xl, bursts)


@jit(nopython=True)
def poisson(om, D1, D2, T, f, g, dt, X=0, hour=1000, height=np.pi, res=True): 
    """Count the number of 'bursts' or trajectories that reach a given 
       threshold after waiting for the trajectory to pass below the threshold again"""

    n, st = int(T / dt), 0 
    D1_, D2_ = np.sqrt(2*D1), np.sqrt(2*D2) 
    X, sdt = X, np.sqrt(dt) 

    Nb = 300 
    Pl = np.zeros(Nb)
    del_x = 50/Nb

    temp, no, dist = 0, [0,0,0], []
    norm = 1/(del_x*(n-st))

    time = 0

    for i in range(n): 
        inc = dt * f(X, om) + sdt * g(X, D1_) * np.random.randn() + \
                                    sdt * D2_ * np.random.randn()
        X += inc
                                             
        time += dt

        if ((X >= height) and (X-inc < height) and time > 1000*dt):
            temp += 1
            dist.append(time)
            time = 0

            if res==True: X = 0

    return(no, dist, Pl)

def num_burst(D_dm, om, Dadd, f_, g, dt, T, a2=0, n_=0, m_=[1], init=0, ref=False, 
              a=-np.pi/2, b=np.pi/2, act='res', save=False, f=None, ax=None, num=0):
    """Numerical bursts on the right side"""

    b_dm, b2_dm = [], []

    for i in D_dm:  
        for j in Dadd: 
            for m in m_:
                tl, X, b_ = burst_counter(om, i, j, T, f_, g, dt, a2=a2, n_=n_, m_=m, 
                                         X0=init, a=a, b=b, act=act) 
                b_dm.append(b_[0]/T); b2_dm.append(b_[1]/T) 

    if not ax:
        sns.set_theme(style="white", rc=params)
        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        lab1, lab2 = 'sin. num.', 'sin. an.'
    else: lab1, lab2 = None, None

    b_dm_num = np.array(b_dm) + np.array(b2_dm)
    ol = ['bo', 'go', 'co', 'mo', 'ro']

    if (len(D_dm) > len(Dadd)) and (len(D_dm) > len(m_)):
        ax.plot(D_dm, b_dm_num, ol[num], markersize=5, mec = 'k', label=lab1)
        ax.set_xlabel(r'$D_1$')
    elif (len(m_) > len(Dadd)) and (len(m_) > len(D_dm)):
        ax.plot(m_, b_dm_num, ol[num], markersize=5, mec = 'k', label=lab1)
        ax.set_xlabel(r'$m$')
        D_dm = m_
    else:
        ax.plot(Dadd, b_dm_num, ol[num], markersize=5, mec = 'k', label=lab1)
        ax.set_xlabel(r'$D_2$')
        D_dm = Dadd

    if ref:
        ax.plot(D_dm, b_dm, 'ro', markersize=5, mec = 'k', label='sin. num. abs.')

    ax.legend()

    if save: 
        f.savefig('/'+save+'.png', bbox_inches='tight', format='png', dpi=120)

    return(f, ax)


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth data with a Savitzky-Golay filter"""
    from math import factorial
       
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError: #, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def smooth_data_fft(arr, span): 
    """Smooth data using Fourier transformation"""
    
    from scipy.fftpack import rfft, irfft
    w = rfft(arr)
    spectrum = w ** 2
    cutoff_idx = spectrum < (spectrum.max() * (1 - np.exp(-span / 2000)))
    w[cutoff_idx] = 0
    return irfft(w)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
 
def num_an_burst(D_dm, om, Dadd, f_, g, dt, T, n_=0, m_=[1], a2=0, init=0,  
                 an='math', a=-np.pi/2, b=np.pi/2, act='ref', 
                 leg='Dm', save=False, f=None, ax=None, num=0, 
                 log=False, legend=False, tit="om1_Dm_Dadd", norm=1): 
    """Numerical and analytical bursts on the right side"""

    l1 = [D_dm, Dadd, m_] 
    l2 = [len(D_dm), len(Dadd), len(m_)] 
    max_ = l1[l2.index(max(l2))] 
    
    b_dm, b2_dm = [], []

    for i in D_dm:  
        for j in Dadd: 
            for m in m_: 
                tl, X, b_ = burst_counter(om, i, j, T, f_, g, dt, a2=a2, n_=n_, 
                                          m_=m, X0=init, a=a, b=b, act=act) 
                b_dm.append(b_[0]/T); b2_dm.append(b_[1]/T)

    st = "/"
    if an=='simp': 
        df=pd.read_csv(st+"c++/DATA/anal_FMPT_"+tit+".dat", sep=" ", header=None)
        df = df.dropna(axis='columns', how='all')
    elif an=='math': 
        df=pd.read_csv(st+"mathematica/plots/fmpt_mon_n0_m_b1_Dm_Dadd.txt", sep='\t', header=None)
        df = df.dropna(axis='columns', how='all')
    else:
        df = pd.DataFrame({'x' : [], 'D1' : [], 'D2' : [], 'm' : [], 'Tn' : []})

    df.columns=['x', 'D1', 'D2', 'm', 'Tn']

    b_dm_an = []

    for i in D_dm:
        for j in Dadd:
            #for m in m_:
            df2 = df[df['D1'] == i]
            df2 = df2[df2['x'] == 0]
            df2 = df2[df2['D2'] == j] 
            if an=='math': df2 = df2[round(df2['m'],1) == round(m_[0],1)]

            if a==0:  
                if np.array(df2['m']): 
                    Tn = 1/np.array(df2['m'])[0]
                    b_dm_an.append(norm*Tn)
                else: b_dm_an.append(np.nan)
            else:
                if np.array(df2['Tn']):
                    Tn = 1/np.array(df2['Tn'])[0]
                    b_dm_an.append(Tn)
                else: b_dm_an.append(np.nan)

    if not ax:
        s = 1.0
        params = {'xtick.labelsize': s*20, 'ytick.labelsize': s*20,
              'axes.labelsize': s*20, 'axes.titlesize': s*20,  
              'font.size': s*20, 'legend.handlelength': 2}

        sns.set_theme(style="white", rc=params)
        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        lab1, lab2 = 'sin. num.', 'sin. an.'
    else: 
        lab1, lab2 = None, None

    b_dm_num = np.array(b_dm) 
    ol = ['bo', 'g>', 'cs', 'm8', 'rd']
    ll = ['b-', 'g-', 'c-', 'm-', 'r-']

    if leg=='m': lab1=r"$m = {}$".format(m_[0]) 
    if leg=='Dadd': lab1=r"$D_{add} =$"+"{}".format(Dadd[0]) 
    if leg=='Dm': lab1=r"$D_m = {}$".format(D_dm[0])
    if leg=='n': lab1=r"$n = {}$".format(n_)
 
    if log: 
        ax.loglog(max_, b_dm_num, ol[num], markersize=8, mec = 'k', label=lab1) 
    else:
        ax.plot(max_, b_dm_num, ol[num], markersize=9, mec = 'k', label=lab1)

    if D_dm == max_: ax.set_xlabel(r'$D_m$')
    elif m_ == max_: ax.set_xlabel(r'$m$')
    else: ax.set_xlabel(r'$D_2$')
    ax.set_ylabel(r'$T^{-1}$') 

    ax.tick_params(bottom=True, left=True)
    ax.tick_params(direction="in", length=4, width=1, grid_alpha=0.5)

    if b_dm_an: ax.plot(max_, b_dm_an, ll[num]) 

    if legend: ax.legend(fontsize=14, ncol=2, framealpha=0.5)

    if save: 
        f.savefig('/'+save+'.png', bbox_inches='tight', format='png', dpi=140)

    return(f, ax)

def num_burst_(ax, D_dm, m_=[1], b=np.pi/2, file='num_burst_om1_m1_T10000_dtp000002_xinit0p1', sep='\t'): 
    """Numerical and analytical bursts on the right side"""
    
    s = '/Users/tphillips/multiplicative_noise/sims/'
    df=pd.read_csv(s+file+'.csv', sep=sep) # ';'

    n_ = [0.0, 0.2, 0.4, 0.6]
    ol = ['bo', 'g>', 'cs', 'm8', 'rd']

    for j in range(len(n_)):
        b_dm = []

        for i in D_dm:  
            for m in m_: 
                df2 = df[(round(df['D1'],1)==round(i,1)) & (round(df['m'],1)==round(m,1))  
                        & (round(df['n'],1)==round(n_[j],1)) & (round(df['xc'],1)==round(b,1))] 
                b_dm.append(np.array(df2['x'])[0]) 

        b_dm_num = np.array(b_dm) 
        ax.loglog(D_dm[::2], b_dm_num[::2], ol[j], markersize=8, mec = 'k', label=r'$n =$ {}'.format(n_[j])) 

    return(ax)

def an_burst(f, ax, save=False, norm=1, i=0):
    """Numerical and analytical bursts on the right side"""

    st = "/"
    l = ["Dm_n0.6_m1_b4_2.txt", "Dm_n0.6_m2_b4.txt", "Dm_n0.6_m1p5_b4.txt", 
         "Dm_n0.6_m0p8_b4.txt", "Dm_n0.6_m1_b10.txt", "Dm_n0.6_m1_b30.txt",
         "Dm_n0.6_m1_b4_xmin0p01.txt", "Dm_n0.6_m1_b4_xmin0p002.txt", 
         "Dm_n0.6_m1_b4_xmin0p001.txt"]
    df=pd.read_csv(st+"T_mon_closed_"+l[i], sep=",", header=None)
    df.columns=['d', 'Tn0', 'Tn02', 'Tn04', 'Tn06']
    df = df.dropna(axis='columns', how='all')

    ll = ['b-', 'g-', 'c-', 'm-', 'r-']

    s = 4
    ax.loglog(df['d'][s:], norm*1/np.array(df['Tn0'])[s:],  ll[0]) 
    ax.loglog(df['d'][s:], norm*1/np.array(df['Tn02'])[s:], ll[1]) 
    ax.loglog(df['d'][s:], norm*1/np.array(df['Tn04'])[s:], ll[2]) 
    if i != 3: ax.loglog(df['d'][s:], norm*1/np.array(df['Tn06'])[s:], ll[3]) 

    ax.tick_params(bottom=True, left=True) 
    ax.tick_params(direction="in", length=4, width=1, grid_alpha=0.5) 

    if save: 
        f.savefig('/'+save+'.png', bbox_inches='tight', format='png', dpi=120)

    return(f, ax)

def num_an_burst2(D_dm, om, Dadd, dt, T, init=0,
                 save=False, f=None, ax=None, num=0):
    """bursts on both the left and right sides"""

    b_dm, b2_dm = [], []

    for i in D_dm:  
        for j in Dadd:
            tl, X, b = burst_counter(om, i, j, T, f_, g, dt, X=init) 
            b_dm.append(b[0]); b2_dm.append(b[1]) 

    if not ax:
        sns.set_theme(style="white", rc=params)
        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        lab1, lab2 = 'right side', 'left side'
    else: lab1, lab2 = None, None

    ol = ['bo', 'go', 'co', 'mo', 'ro']
    ll = ['b', 'g', 'c', 'm', 'r']

    if len(D_dm) > len(Dadd):
        ax.plot(D_dm, b_dm, ol[num], markersize=7, mec = ll[num], label=lab1)
        ax.plot(D_dm, b2_dm, ol[num], markersize=7, mec = ll[num], label=lab2)
        ax.plot(D_dm, b_dm, 'k-')
        ax.plot(D_dm, b2_dm, 'r-')
        ax.set_xlabel(r'$D_1$')
    else:
        ax.plot(Dadd, b_dm, ol[num], markersize=7, mec = ll[num], label=lab1)
        ax.plot(Dadd, b2_dm, ol[num], markersize=7, mec = ll[num], label=lab2)
        ax.plot(Dadd, b_dm, 'k-')
        ax.plot(Dadd, b2_dm, 'r-')
        ax.set_xlabel(r'$D_2$')
        D_dm = Dadd

    ax.legend()

    if save: 
        f.savefig('/'+save+'.png', bbox_inches='tight', format='png', dpi=120)

    return(f, ax)

@jit(nopython=True)
def burst_hist(om, D1, D2, T, f, g, dt, n=0, m=1, X_in=0, height=np.pi): 
    """Obtain distribution of the laminar lengths"""

    n_ = int(T / dt)
    D1_, D2_ = np.sqrt(2*D1), np.sqrt(2*D2) 
    X, sdt = X_in, np.sqrt(dt) 

    dist = []
    time = 0

    for i in range(n_): 
        inc = dt * f(X, om, n) + sdt * g(X, D1_, m) * np.random.randn() + \
                                    sdt * D2_ * np.random.randn()

        X += inc                                 
        time += dt

        if (X >= height):
            dist.append(time)
            X = X_in
            time = 0

    return(dist)

@jit(nopython=True)
def burst_hist2(om, D1, D2, n_, f, g, dt, n=0, m=1, X_in=0, height=np.pi): 
    """Same as burst_hist, but the output list has fixed size"""

    D1_, D2_ = np.sqrt(2*D1), np.sqrt(2*D2) 
    X, sdt = X_in, np.sqrt(dt) 

    dist = []
    time = 0

    while len(dist) < n_: 
        inc = dt * f(X, om, n) + sdt * g(X, D1_, m) * np.random.randn() + \
                                    sdt * D2_ * np.random.randn()
        X += inc                                 
        time += dt

        if ((X < height) and (X-inc >= height)):
            time = 0
        elif ((X >= height) and (X-inc < height)):
            dist.append(time)

    return(dist)

def exp_surv(x,a,lam): return(a*(np.exp(-lam*x))) 

def exp_dec_dist(Dm, Da, f_, g_, n=0, m=1, T=5000000, X=0, height=np.pi/4, 
                 dt=0.001, delT=2000000, plot=True):
    """Calculate cumulative distribution of bursts"""

    y, T2, delT = [], 0, 2000000
    while len(y) < 1000 and T2<5*delT: 
        y += burst_hist(1, Dm, Da, T+T2, f_, g_, n=n, X_in=X, m=m, dt=dt, height=height)
        y = [x for x in y if x > 2.8]
        
        T2 += delT
        print(T2, len(y))
        if len(y) < 2000: print('len y = ', len(y))

    try:
        n_, bins = np.histogram(y, density=True, bins=200) 
        y_cum = np.cumsum(n_)/np.max(np.cumsum(n_))

        y_surv = 1-y_cum
        x = bins[1:]

        popt, pcov = scipy.optimize.curve_fit(exp_surv, x, y_surv) 

        if plot:
            f, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(x, y_surv, '.')
            ax.plot(bins[:-50], exp_surv(bins[:-50], *popt), 'r', label=r'$p(x) \sim e^{-bx}$' 
                                        +'\n'+r'$b$={}'.format(round(popt[1],2))) 

        return(popt[1])
    
    except ValueError:
        print("T too short - Dm:", Dm, " Da:", Da, " T:", T)
        return(np.nan)
    

def lam_list(Dm=np.arange(0.5,10,0.5), Da=0, n=0, m=1, T=500000, dt=0.001):

    Dl, lam_l = Dm, []

    for D in Dl:
        lam = exp_dec_dist(D, Da, f_mon, g_mon, n=n, m=m, T=T, dt=dt, plot=False)
        lam_l.append(lam)

        clear_output(wait=True)
        print(lam)
    
    df = pd.DataFrame({'D':Dl, 'lam':lam_l})
    df.to_csv('exp_par_Dm_Da'+str(Da)+'_m'+str(m)+'_n'+str(n)+'_T'
              +str(T)+'_dt'+str(dt)+'.csv', sep='\t')

    return(df)


def m_mom_gen(a, a2, D, D2, T, dt, n=0, m=1, vp=[0.5,0.4], r=30, plot=True, lim=10):
    """Obtain a list of maxima and FWHM for a range of parameters"""

    varl, fwhm, max_l = [], [], []
    klist = np.zeros(40)
    
    for i in range(r):
        clear_output(wait=True)
        print(i)

        var_ = vp[0] + vp[1]*i

        if a=='var':
            x_per, P_per = stat_den_sde_gen(var_, a2, D, D2, T, f_mon,  
                                g_mon, dt, osc=False, lim=lim, n=n, m=m)
        elif a2=='var':
            x_per, P_per = stat_den_sde_gen(a, var_, D, D2, T, f_mon, 
                                g_mon, dt, osc=False, lim=lim, n=n, m=m)
        elif D=='var':
            x_per, P_per = stat_den_sde_gen(a, a2, var_, D2, T, f_mon,  
                                g_mon, dt, osc=False, lim=lim, n=n, m=m)  
        elif D2=='var':
            x_per, P_per = stat_den_sde_gen(a, a2, D, var_, T, f_mon, 
                                g_mon, dt, osc=False, lim=lim, n=n, m=m)
        elif n=='var':
            x_per, P_per = stat_den_sde_gen(a, a2, D, D2, T, f_mon, 
                                g_mon, dt, osc=False, lim=lim, n=var_, m=m)
        elif m=='var':
            x_per, P_per = stat_den_sde_gen(a, a2, D, D2, T, f_mon, 
                                g_mon, dt, osc=False, lim=lim, n=n, m=var_)
            
        varl.append(var_)
        lower_x, upper_x = half_max_x(x_per, P_per)

        if D=='var':
            if i > 2 and (upper_x - lower_x)>20:
                fwhm.append(lim - (upper_x - lower_x))
            else:
                fwhm.append(upper_x - lower_x)
        else:
           fwhm.append(upper_x - lower_x) 

        xmax = find_max(x_per, P_per) 
        max_l.append(xmax)

        if plot:
            f, ax = plt.subplots(1, 1, figsize=(4, 2))
            ax.plot(x_per, P_per, 'k-')
            ax.axvline(lower_x, color='grey'); ax.axvline(upper_x, color='grey')
            ax.axvline(xmax, color='r')

    return(varl, fwhm, max_l)

def max_gen(D, om, sig, n=0, m=1, nu=0):
    gam = 1/(n-2*m+1)

    if ((2-nu)*m*D - sig) == 0:
        return(np.nan)
    else:     
        xm = (((2-nu)*m*D - sig)/om)**gam
        return(xm)
    

@jit(nopython=True)
def stat_den_strat_s(om, D1, D2, T, f_, g, dt, osc=False, Nb=300, 
                     lim=np.pi, xinit=0.1, n=0, m=1, start=50000, sig=0): 
    """Stationary probability function considering Stratonovich calculus"""

    n_ = int(T / dt)  
    st = start
    Pl = np.zeros(Nb) 
    D1_, D2_ = np.sqrt(2*D1), np.sqrt(2*D2)
    sdt = np.sqrt(dt)
    X = xinit 

    xm = -lim       
    del_x = 2*lim/Nb
    x = np.arange(-lim,lim,del_x)
    X = 2*lim*np.random.rand() - lim
    norm = 1/(del_x*(n_-st))

    for i in range(n_):

        X += dt * (f_(X, om, sig) + 0.5*D1_*np.cos(X)*g(X, D1_, m)) + \
                sdt * g(X, D1_, m) * np.random.randn() + \
                                sdt * D2_ * np.random.randn()

        X = (X + 2*lim)%(2*lim) 
        if(X >= lim): X -= 2*lim 
        if math.isnan(X): X = 2*lim*np.random.rand() - lim

        if i > st: 
            k = int(np.floor((X-xm)/del_x)) 
            Pl[k] += norm  

    return(x, Pl)

@jit(nopython=True)
def stat_den_strat_l(om, D1, D2, T, f_, g, dt, Nb=300, low=0, lim=np.pi, 
                     xinit=0.1, m=1, start=50000, sig=0, lin='gen'): 
    """Calculate stationary probability density for some specific functions"""

    n_ = int(T / dt)  
    st = start
    Pl = np.zeros(Nb) 
    D1_, D2_ = np.sqrt(2*D1), np.sqrt(2*D2)
    sdt = np.sqrt(dt)
    X = xinit 
  
    b = lim 
    del_x = abs(b-low)/Nb     
    x = np.arange(low,b,del_x)
    X = xinit 
    norm = 1/(del_x*(n_-st))

    if lin=='gen':
        for i in range(n_):
            
            X += dt * (f_(X, om, sig) + 0.5*D1_*g(X, D1_, m)) + \
                 sdt * g(X, D1_, m) * np.random.randn() + \
                                sdt * D2_ * np.random.randn()

            if i > st:  
                k = int(np.floor((X)/del_x)) 
                if (abs(k) < Nb):
                    Pl[k + int(abs(low)/del_x)] += norm   
    if lin=='zero':
        for i in range(n_):
            
            X += dt * (om + (D1-sig)*X) + \
                 sdt * D1_ * X * np.random.randn() + \
                                sdt * D2_ * np.random.randn()

            if i > st:  
                k = int(np.floor((X)/del_x)) 
                if (abs(k) < Nb):
                    Pl[k + int(abs(low)/del_x)] += norm 
    else:
        for i in range(n_):
            
            X += dt * (om + (D1+sig)*X) + \
                 sdt * D1_ * X * np.random.randn() + \
                                sdt * D2_ * np.random.randn()

            if X > 10:
                X = 10 - abs(X-10)/2
            if i > st:  
                k = int(np.floor((X)/del_x)) 
                if (abs(k) < Nb):
                    Pl[k + int(abs(low)/del_x)] += norm 

    return(x, Pl)

###

def max_fwhm(om0=1, sig=-1, D1_inc=0.4, D2=0, T=10000, dt=0.001, 
             dt_l=0.001, r=21, D0=0, v_sig=False, show=True, nu=1, lim=np.pi):
    """Calculate maximum and FWHM"""
    
    Dlist = []
    p_max4, p_max_lin = [], []
    fwhm_l, fwhm_, max_an = [], [], []
    sig_ = sig

    for i in range(1,r): 
        D = D0
        om = om0 + D1_inc*i
        Dlist.append(om)

        if v_sig: sig_ = -D + sig

        clear_output(wait=True); print(i)

        # Sinusoidal
        x_per, P_per = stat_den_strat_s(om, D, D2, T, f_sin, g, dt, osc=True, Nb=1000, sig=sig_) 
        xmax = find_max(x_per, P_per) 
        p_max4.append(xmax-np.pi)

        hp = int(len(x_per)/2)
        lower_x, upper_x = half_max_x(x_per[hp:], P_per[hp:])

        fwhm_.append(upper_x - lower_x)
  
        # Linear
        x_per_l, P_per_l = stat_den_strat_l(om, D, D2, T, f_lin, g_lin, dt_l, 
                                            osc=False, Nb=1000, sig=-sig_, lim=lim, lin='zero')
        xmax_l = find_max(x_per_l, P_per_l)
        p_max_lin.append(xmax_l)

        lower_x_l, upper_x_l = half_max_x(x_per_l, P_per_l)
        fwhm_l.append(upper_x_l - lower_x_l)

        max_an.append(max_an_lin(D, sig_, nu=nu, om=om))

        if show:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2))
            ax1.plot(x_per, P_per, 'r-')
            ax1.axvline(lower_x); ax1.axvline(upper_x); ax1.axvline(xmax)
            ax2.plot(x_per_l, P_per_l, 'r.')
            ax2.axvline(lower_x_l); ax2.axvline(upper_x_l); ax2.axvline(xmax_l)
            ax2.set_xlim([0, np.pi])

    p_max_osc = np.array(p_max4)+np.pi
    for i in range(5):
      p_max_osc[3+i] += 0 

    return(Dlist, p_max_osc, p_max_lin, fwhm_l, fwhm_, max_an)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_max_fwhm(Dlist, p_max_osc, p_max_lin, fwhm_l, fwhm_, 
                  m_an, save=None, sh_lin=True, s_=1, an_sin=False):
    """Plot maximum and FWHM"""
   
    s = 0.9
    params = {'xtick.labelsize': s*20, 'ytick.labelsize': s*20,
                'axes.labelsize': s*20, 'axes.titlesize': s*20,  
                'font.size': s*20, 'legend.handlelength': 2}

    sns.set_theme(style="white", rc=params)

    p_max_osc_clean = p_max_osc.copy()

    f, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.6))

    if sh_lin:
        ax.plot(Dlist, p_max_lin, 'bo', markersize=7, mec = 'k') 
        ax.plot(Dlist, m_an, 'b-', label='Lin. (An.)')
        s_lab = 'Sin. (An.)' 
        s_lab2 = 'Sin. (Num.)'
        ax.set_ylabel(r'$\theta_{max}, x_{max}$')
    else:
        s_lab  = 'An.'
        s_lab2 = 'Num.'
        ax.set_ylabel(r'$\theta_{max}$')

    if an_sin==True:
        df=pd.read_csv("/c++/DATA/trial10.dat",sep=" ",header=None)
        df = df.dropna(axis='columns',how='all')
        df.columns=['D1','max']
        ax.plot(df['D1'][1:], df['max'][1:], 'g-', label=s_lab)

    c = p_max_osc_clean[4]/m_an[4]
    ax.plot(Dlist, c*np.array(m_an), 'g-', label=s_lab)
    ax.plot(Dlist, p_max_osc_clean, 'go', markersize=7, mec = 'k')

    ax.legend(fontsize=17)
    ax.set_xlabel(r'$\omega$')

    ax.tick_params(bottom=True, left=True)
    ax.tick_params(direction="in", length=4, width=1, grid_alpha=0.5)

    s_lin = '_'

    if sh_lin:
        c = fwhm_l[7]/m_an[7]
        ax2.plot(Dlist, c*np.array(m_an), 'b-')
        ax2.plot(Dlist, fwhm_l, 'bo', markersize=7, mec = 'k', label='Lin. (Num.)')
        s_lin = '_lin_'
    else:
        s_lab = None

    c = fwhm_[4]/m_an[4]
    ax2.plot(Dlist, c*np.array(m_an), 'g-')
    ax2.plot(Dlist, fwhm_, 'go', markersize=7, mec = 'k', label=s_lab2)
    ax2.set_xlabel(r'$\omega$')
    ax2.set_ylabel(r'FWHM')

    ax2.tick_params(bottom=True, left=True)
    ax2.tick_params(direction="in", length=4, width=1, grid_alpha=0.5)
    if s_lab: ax2.legend(fontsize=17)

    f.tight_layout()

    if save: 
        f.savefig('/max_fwhm'+s_lin+'sin_'+save+'.png', bbox_inches='tight', format='png', dpi=160*s_)

def max_an_lin(D, sig, nu=1, om=1):
    den = (2-nu)*D - sig
    if den == 0:
        return(np.nan)
    else:
        return(om/den)

@jit(nopython=True)
def g_d(X, D, m=None):
    """Diffusion function with cosine term"""
    return(D * np.cos(X))

@jit(nopython=True)
def g_lin_d(X, D, m=None):
    """Constant diffusion term"""
    return(D)

def plot_stat_dis(x_osc, P_osc, x_lin, P_lin, ax=None, f=None,
                  om=1, D1=0.9, D2=0, T=10000, sig=-1, lin=True, 
                  ylim=None, s_=1, lin_pi=None):
    """Plot stationary distribution"""

    params = {'xtick.labelsize': 19, 'ytick.labelsize': 19,
              'axes.labelsize': 21, 'axes.titlesize': 21,  
              'font.size': 21, 'legend.handlelength': 2}

    sns.set_theme(style="white", rc=params)

    if f: 
        a = 1
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 6))
        a = 0

    P_lin_ = np.array(P_lin)
    ax[a,0].plot(x_osc, P_osc, 'k', label='Sin.') 
    if lin: ax[a,0].plot(x_lin, P_lin_, 'r', label='Lin.')
    if lin_pi: 
        l_pi_2 = np.array(lin_pi[0]) - np.pi #2*np.pi
        ax[a,0].plot(l_pi_2, lin_pi[1], 'r', label='Lin.')
        ax[a,0].plot(l_pi_2[::10], lin_pi[1][::10], 'ro', markersize=5, mec = 'k')

    df=pd.read_csv("/c++/DATA/stat_nFPE_om1_Dm2_Dadd0.04.dat",sep=" ",header=None)
    df = df.dropna(axis='columns',how='all')
    df.columns=['phi', 'D1', 'D2', 'P', 'Pn', 'I', 'PdB']
    y = np.array(df['Pn'])
    x = np.arange(-np.pi, np.pi, 2*np.pi/len(y))
    ax[a,0].plot(x_osc[::5], P_osc[::5], 'k+', markersize=5, mec = 'k')

    if lin: 
        print("linear_Dm"+str(D1)+"_sigm1.txt")
        #df=pd.read_csv("/linear_Dm"+str(D1)+"_sigm1.txt", sep='\t', header=None)
        #df.columns=['phi', 'P']
        #df = df.sort_values(by=['phi'])
        scale = np.max(P_lin_)/np.max(df['P'])
        #ax[a,0].plot(df['phi'][::10], scale*np.array(df['P'][::10]), 'ro', markersize=5, mec = 'k')

        ax[a,0].plot(x_lin[::10], P_lin_[::10], 'ro', markersize=5, mec = 'k')
    
    ax[a,0].tick_params(bottom=True, left=True)
    ax[a,0].tick_params(direction="in", length=4, width=1, grid_alpha=0.5)
    ax[a,0].set_title(r"$D_m = {}$".format(D1), fontsize=19)
    labels(ax[a,0])
    ax[a,0].set_xlim([-np.pi, np.pi])
    if ylim: ax[a,0].set_ylim(ylim)

    dt = 0.000001
    T = 20

    
    tl, xl = SDE_ev(om, D1, D2, T, f_sin, g, dt, xinit=1, osc=True, meth='strat', g_der=g_d, n=sig)
    tl, detlist = SDE_ev(om, 0, 0, T, f_sin, g, dt, xinit=1, osc=True, meth='strat', g_der=g_d, n=sig)

    ax[a,1].plot(tl, xl, '.', markersize=0.6)
    ax[a,1].plot(tl, detlist, 'r.', markersize=0.2)
    ax[a,1].set_yticks([-np.pi, 0, np.pi])
    ax[a,1].set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
    ax[a,1].set_xlabel(r'$t$')
    ax[a,1].set_ylabel(r'$\theta$')
    ax[a,1].tick_params(bottom=True, left=True)
    ax[a,1].tick_params(direction="in", length=4, width=1, grid_alpha=0.5)

    if f != None:
        f.tight_layout()
        save = 'stat_d_sin_lin_strat_flag_Dm{}_Dadd{}_sig{}_dt{}_T{}_om{}'.format(D1, D2, sig, dt, T, om)
        f.savefig('/'+save+'.png', bbox_inches='tight', format='png', dpi=150*s_)
        f.savefig('/'+save+'_2.png', bbox_inches='tight', format='png', dpi=90*s_)
    
    return(f, ax)





