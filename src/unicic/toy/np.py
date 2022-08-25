#!/usr/bin/env python3
'''The "toy" problem for testing, implemented for numpy/cupy

A *_scalar() function takes and produces non-batched input.

A *_batched() function takes input array(s) which are batched over
axis=0 and returns likewise.

'''

import numpy
xp = numpy

Random = numpy.random.default_rng


def predict_scalar(q, ms, xp=xp):
    '''Return a predicted measure at parameter point q on measure
    linspace ms in the context of array module xp.

    q is shaped (3,)

    Return is shaped as ms.
    '''
    mu,sig,mag = q.T            # same for scalar or batched

    gnorm = mag / (xp.sqrt(2*xp.pi))
    
    d = ms - mu             # (nbins,) 
    return gnorm * xp.exp(-0.5*((d/sig)**2))

def predict_batched(q, ms, xp=xp):
    '''Return a predicted measure at parameter point q on measure
    linspace ms in the context of array module xp.

    q is shaped (nbatch,3)

    Return is shaped as (nbatch, ms.size)
    '''
    mu,sig,mag = q.T            # same for scalar or batched

    gnorm = mag / (xp.sqrt(2*xp.pi))
    
    d = ms.reshape(1,-1) - mu.reshape(-1,1) # (nbatch,nbins)
    return gnorm.reshape(-1, 1) * xp.exp(-0.5*((d/sig.reshape(-1,1))**2))


def fluctuate(Npred, rng, xp=xp):
    '''
    Return fluctuated measure of expectation
    '''
    return rng.poisson(Npred)


def statvar_cnp_diag(Npred, N, xp=xp):
    '''Return the diagonal of the statistical covariance matrix.

    Npred may be scalar shape (nbins,) or batched (nbatch, nbins).

    The "combined Neyman-Pearson" construction is used with protection
    for zeros and infinities.'''

    num = 3.0 * N * Npred
    den = 2.0 * N + Npred

    good_N = N > 0                  # "data"
    num = xp.where(good_N, num, 0.5*Npred)
    den = xp.where(good_N, den, 1.0)

    good_both = xp.logical_and(Npred > 0, good_N)
    num = xp.where(good_both, num, N)
    den = xp.where(good_both, den, 1)

    diag = num/den
    return diag

def statvar_cnp_scalar(Npred, N, xp=xp):
    diag = statvar_cnp_diag(Npred, N, xp)
    return diag * xp.eye(N.size)
def statvar_cnp_batched(Npred, N, xp=xp):
    diag = statvar_cnp_diag(Npred, N, xp)
    diag = xp.expand_dims(diag, axis=1)
    return diag * xp.eye(N.size)



