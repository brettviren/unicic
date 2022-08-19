#!/usr/bin/env python3

import jax.numpy as jnp
from jax import random

# measurement space
nbins = 100
xmin = 0
xmax = 10
ms = jnp.linspace(0,10,100,False)

# parameter space

ps = (
    jnp.linspace(0, 10, 100, False), # mu
    jnp.linspace(0, 10, 100, False), # sig
    jnp.linspace(0, 10, 100, False)  # mag
)

mg = jnp.meshgrid(*ps)


def random_parameter(key):
    '''
    Return a random parameter array from the mesh grid
    '''
    ndim = len(ps)
    vmin = jnp.array([0]*ndim)
    vmax = jnp.array([len(d) for d in ps])
    ind = random.randint(key, (ndim,), vmin, vmax).to_py()
    slices = tuple([slice(ind[i], ind[i]+1) for i in range(ndim)])
    return jnp.array([mg[i][slices].squeeze() for i in range(ndim)])

# true gaussian parameters (mu,sigma,magnitude)
# tru = np.array([4.5, 1.1, 100])

def predict(q):
    '''
    The model
    '''
    mu = q[0]
    sig = q[1]
    mag = q[2]

    gnorm = mag / (jnp.sqrt(2*jnp.pi))
    d = ms - mu
    return gnorm * jnp.exp(-0.5*(d/sig)**2)

def most_likely(N):
    '''
    Return parameter that best predicts N.
    '''
    
