# This code is used to obtain animations of the time evolution of the Probability distribution 
# Based on code from johnaparker

import importlib
import numpy as np
import enum
from inspect import getfullargspec

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from scipy.constants import k
from scipy import constants
from scipy.linalg import expm
from scipy import sparse
from scipy.sparse.linalg import expm, eigs, expm_multiply
from scipy.interpolate import RegularGridInterpolator

def value_to_vector(value, ndim, dtype=float):
    """convert a value to a vector in ndim"""
    value = np.asarray(value, dtype=dtype)
    if value.ndim == 0:
        vec = np.asarray(np.repeat(value, ndim), dtype=dtype)
    else:
        vec = np.asarray(value)
        if vec.size != ndim:
            raise ValueError(f'input vector ({value}) does not have the correct dimensions (ndim = {ndim})')

    return vec

def slice_idx(i, ndim, s0):
    """return a boolean array for a ndim-1 slice along the i'th axis at value s0"""
    idx = [slice(None)]*ndim
    idx[i] = s0

    return tuple(idx)

def combine(*funcs):
    """combine a collection of functions into a single function (for probability, potential, and force functions)"""
    def combined_func(*args):
        values = funcs[0](*args)
        for func in funcs[1:]:
            values += func(*args)

        return values

    return combined_func

class boundary(enum.Enum):
    """enum for the types ofboundary conditions"""
    reflecting = enum.auto()
    periodic   = enum.auto()

def vectorize_force(f):
    """decorator to vectorize a force function"""
    ndim = len(getfullargspec(f).args)
    signature = ','.join(['()']*ndim)
    signature += '->(N)'

    vec_f = np.vectorize(f, signature=signature)
    def new_func(*args):
        return np.rollaxis(vec_f(*args), axis=-1, start=0)

    return new_func

###

class fokker_planck:
    def __init__(self, *, temperature, drag, extent, resolution,
            potential=None, force=None, boundary=boundary.reflecting): # fplanck.boundary.reflecting

        self.extent = np.atleast_1d(extent)
        self.ndim = self.extent.size

        self.temperature = value_to_vector(temperature, self.ndim)
        self.resolution  = value_to_vector(resolution, self.ndim)

        self.potential = potential
        self.force = force
        self.boundary = value_to_vector(boundary, self.ndim, dtype=object)

        self.beta = 1/(constants.k*self.temperature)

        self.Ngrid = np.ceil(self.extent/resolution).astype(int)
        axes = [np.arange(self.Ngrid[i])*self.resolution[i] for i in range(self.ndim)]
        for axis in axes:
            axis -= np.average(axis)
        self.axes = axes
        self.grid = np.array(np.meshgrid(*axes, indexing='ij'))

        self.Rt = np.zeros_like(self.grid)
        self.Lt = np.zeros_like(self.grid)
        self.potential_values = np.zeros_like(self.grid[0])
        self.force_values = np.zeros_like(self.grid)

        self.drag = np.zeros_like(self.grid)
        self.diffusion = np.zeros_like(self.grid)
        if callable(drag):
            self.drag[...] = drag(*self.grid)
        elif np.isscalar(drag):
            self.drag[...] = drag
        elif isinstance(drag, Iterable) and len(drag) == self.ndim:
            for i in range(self.ndim):
                self.drag[i] = drag[i]
        else:
            raise ValueError(f'drag must be either a scalar, {self.ndim}-dim vector, or a function')

        self.mobility = 1/self.drag
        for i in range(self.ndim):
            self.diffusion[i] = constants.k*self.temperature[i]/self.drag[i]

        if self.potential is not None:
            U = self.potential(*self.grid)
            self.potential_values += U
            self.force_values -= np.gradient(U, *self.resolution)

            for i in range(self.ndim):
                dU = np.roll(U, -1, axis=i) - U
                self.Rt[i] += self.diffusion[i]/self.resolution[i]**2*np.exp(-self.beta[i]*dU/2)

                dU = np.roll(U, 1, axis=i) - U
                self.Lt[i] += self.diffusion[i]/self.resolution[i]**2*np.exp(-self.beta[i]*dU/2)

        if self.force is not None:
            F = np.atleast_2d(self.force(*self.grid))
            self.force_values += F

            for i in range(self.ndim):
                dU = -(np.roll(F[i], -1, axis=i) + F[i])/2*self.resolution[i]
                self.Rt[i] += self.diffusion[i]/self.resolution[i]**2*np.exp(-self.beta[i]*dU/2)

                dU = (np.roll(F[i], 1, axis=i) + F[i])/2*self.resolution[i]
                self.Lt[i] += self.diffusion[i]/self.resolution[i]**2*np.exp(-self.beta[i]*dU/2)

        if self.force is None and self.potential is None:
            for i in range(self.ndim):
                self.Rt[i] = self.diffusion[i]/self.resolution[i]**2
                self.Lt[i] = self.diffusion[i]/self.resolution[i]**2

        for i in range(self.ndim):
            if self.boundary[i] == boundary.reflecting:
                    idx = slice_idx(i, self.ndim, -1)
                    self.Rt[i][idx] = 0

                    idx = slice_idx(i, self.ndim, 0)
                    self.Lt[i][idx] = 0
            elif self.boundary[i] == boundary.periodic:
                    idx = slice_idx(i, self.ndim, -1)
                    dU = -self.force_values[i][idx]*self.resolution[i]
                    self.Rt[i][idx] = self.diffusion[i][idx]/self.resolution[i]**2*np.exp(-self.beta[i]*dU/2)

                    idx = slice_idx(i, self.ndim, 0)
                    dU = self.force_values[i][idx]*self.resolution[i]
                    self.Lt[i][idx] = self.diffusion[i][idx]/self.resolution[i]**2*np.exp(-self.beta[i]*dU/2)
            else:
                raise ValueError(f"'{self.boundary[i]}' is not a valid a boundary condition")

        self._build_matrix()

    def _build_matrix(self):
        """build master equation matrix"""
        N = np.product(self.Ngrid)

        size = N*(1 + 2*self.ndim)
        data = np.zeros(size, dtype=float)
        row  = np.zeros(size, dtype=int)
        col  = np.zeros(size, dtype=int)

        counter = 0
        for i in range(N):
            idx = np.unravel_index(i, self.Ngrid)
            data[counter] = -sum([self.Rt[n][idx] + self.Lt[n][idx]  for n in range(self.ndim)])
            row[counter] = i
            col[counter] = i
            counter += 1

            for n in range(self.ndim):
                jdx = list(idx)
                jdx[n] = (jdx[n] + 1) % self.Ngrid[n]
                jdx = tuple(jdx)
                j = np.ravel_multi_index(jdx, self.Ngrid)

                data[counter] = self.Lt[n][jdx]
                row[counter] = i
                col[counter] = j
                counter += 1

                jdx = list(idx)
                jdx[n] = (jdx[n] - 1) % self.Ngrid[n]
                jdx = tuple(jdx)
                j = np.ravel_multi_index(jdx, self.Ngrid)

                data[counter] = self.Rt[n][jdx]
                row[counter] = i
                col[counter] = j
                counter += 1

        self.master_matrix = sparse.csc_matrix((data, (row, col)), shape=(N,N))

    def steady_state(self):
        """Obtain the steady state solution"""
        vals, vecs = eigs(self.master_matrix, k=1, sigma=0, which='LM')
        steady = vecs[:,0].real.reshape(self.Ngrid)
        steady /= np.sum(steady)

        return steady

    def propagate(self, initial, time, normalize=True, dense=False):
        """Propagate an initial probability distribution in time"""
        p0 = initial(*self.grid)
        if normalize:
            p0 /= np.sum(p0)

        if dense:
            pf = expm(self.master_matrix*time) @ p0.flatten()
        else:
            pf = expm_multiply(self.master_matrix*time, p0.flatten())

        return pf.reshape(self.Ngrid)

    def propagate_interval(self, initial, tf, Nsteps=None, dt=None, normalize=True):
        """Propagate an initial probability distribution over a time interval, return time and the probability distribution at each time-step"""
        p0 = initial(*self.grid)
        if normalize:
            p0 /= np.sum(p0)

        if Nsteps is not None:
            dt = tf/Nsteps
        elif dt is not None:
            Nsteps = np.ceil(tf/dt).astype(int)
        else:
            raise ValueError('specifiy either Nsteps or Nsteps')

        time = np.linspace(0, tf, Nsteps)
        pf = expm_multiply(self.master_matrix, p0.flatten(), start=0, stop=tf, num=Nsteps, endpoint=True)
        return time, pf.reshape((pf.shape[0],) + tuple(self.Ngrid))

    def probability_current(self, pdf):
        """Obtain the probability current of the given probability distribution"""
        J = np.zeros_like(self.force_values)
        for i in range(self.ndim):
            J[i] = -(self.diffusion[i]*np.gradient(pdf, self.resolution[i], axis=i) 
                  - self.mobility[i]*self.force_values[i]*pdf)

        return J

def delta_function(r0):
    """a discrete equivalent of a dirac-delta function centered at r0"""
    r0 = np.atleast_1d(r0)

    def pdf(*args):
        values = np.zeros_like(args[0])

        diff = sum([(r0[i] - args[i])**2 for i in range(len(args))])
        idx = np.unravel_index(np.argmin(diff), diff.shape)
        values[idx] = 1

        return values
        
    return pdf

def gaussian_pdf(center, width):

    center = np.atleast_1d(center)
    ndim = len(center)
    width = value_to_vector(width, ndim)

    def pdf(*args):
        values = np.ones_like(args[0])

        for i, arg in enumerate(args):
            values *= np.exp(-np.square((arg - center[i])/width[i]))

        return values/np.sum(values)

    return pdf

def uniform_pdf(func=None):

    def pdf(*args):
        if func is None:
            values = np.ones_like(args[0])
        else:
            values = np.zeros_like(args[0])
            idx = func(*args)
            values[idx] = 1

        return values/np.sum(values)

    return pdf

def harmonic_potential(center, k):

    center = np.atleast_1d(center)
    ndim = len(center)
    k = value_to_vector(k, ndim)

    def potential(*args):
        U = np.zeros_like(args[0])

        for i, arg in enumerate(args):
            U += 0.5*k[i]*(arg - center[i])**2

        return U

    return potential

def gaussian_potential(center, width, amplitude):

    center = np.atleast_1d(center)
    ndim = len(center)
    width = value_to_vector(width, ndim)

    def potential(*args):
        U = np.ones_like(args[0])

        for i, arg in enumerate(args):
            U *= np.exp(-np.square((arg - center[i])/width[i]))

        return -amplitude*U

    return potential

def uniform_potential(func, U0):

    def potential(*args):
        U = np.zeros_like(args[0])
        idx = func(*args)
        U[idx] = U0

        return U

    return potential

def potential_from_data(grid, data):

    grid = np.asarray(grid)
    if grid.ndim == data.ndim == 1:
        grid = (grid,)

    f = RegularGridInterpolator(grid, data, bounds_error=False, fill_value=None)
    def potential(*args):
        return f(args)

    return potential

def force_from_data(grid, data):

    grid = np.asarray(grid)
    if grid.ndim == data.ndim == 1:
        grid = (grid,)

    f = RegularGridInterpolator(grid, np.moveaxis(data, 0, -1), bounds_error=False, fill_value=None)
    def force(*args):
        return np.moveaxis(f(args), -1, 0)

    return force