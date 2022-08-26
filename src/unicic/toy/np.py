#!/usr/bin/env python3
'''The "toy" problem for testing, implemented for numpy/cupy

A *_scalar() function takes and produces non-batched input.

A *_batched() function takes input array(s) which are batched over
axis=0 and returns likewise.

'''

import numpy
xp = numpy

Random = numpy.random.default_rng

def uniform(rng, low=0.0, high=1.0, size=None):
    return rng.uniform(low, high, size)

# def predict_scalar(q, ms, xp=xp):
#     '''Return a predicted measure at parameter point q on measure
#     linspace ms in the context of array module xp.

#     q is shaped (3,)

#     Return is shaped as ms.
#     '''
#     mu,sig,mag = q.T            # same for scalar or batched

#     gnorm = mag / (xp.sqrt(2*xp.pi))
    
#     d = ms - mu             # (nbins,) 
#     return gnorm * xp.exp(-0.5*((d/sig)**2))

def predict(q, ms, xp=xp):
    '''Return a predicted measure at parameter point q on measure
    linspace ms in the context of array module xp.

    q may be scalar or batch, return follows.
    '''
    mu,sig,mag = q.T            # same for scalar or batched

    gnorm = mag / (xp.sqrt(2*xp.pi))
    
    d = ms.reshape(1,-1) - mu.reshape(-1,1) # (nbatch,nbins)
    pred = gnorm.reshape(-1, 1) * xp.exp(-0.5*((d/sig.reshape(-1,1))**2))
    qdims = len(q.shape)
    if qdims == 1:
        pred = pred.reshape((ms.size,))
    return pred

def fluctuate(Npred, rng, xp=xp):
    '''
    Return fluctuated measure of expectation
    '''
    return rng.poisson(Npred)


def statvar_cnp_diag(Npred, Nmeas, xp=xp):
    '''Return the diagonal of the statistical covariance matrix
    following "combined Nyeman-Pearson" construction with additional
    protection for zeros and infinites.
    
    Npred and Nmeas may be any mix of scalar shape (nbins,) or batched
    (nbatch, nbins).  If either are batched, the return is batched.
    If both are batched, they must be batched the same size and a
    batch-to-batch comparison is make.

    '''

    num = 3.0 * Nmeas * Npred
    den = 2.0 * Nmeas + Npred

    good_meas = Nmeas > 0
    num = xp.where(good_meas, num, 0.5*Npred)
    den = xp.where(good_meas, den, 1.0)

    good_both = xp.logical_and(Npred > 0, good_meas)
    num = xp.where(good_both, num, Nmeas)
    den = xp.where(good_both, den, 1)

    diag = num/den
    return diag

# def statvar_cnp_scalar(Npred, N, xp=xp):
#     diag = statvar_cnp_diag(Npred, N, xp)
#     return diag * xp.eye(N.size)
def statvar_cnp(Npred, N, xp=xp):
    diag = statvar_cnp_diag(Npred, N, xp)
    diag = xp.expand_dims(diag, axis=1)
    I = xp.eye(N.shape[-1])
    return diag * I


def inv(a, xp=xp):
    '''
    Return the multiplicative inverse of a.

    Raises ValueError if singular
    '''
    try:
        return xp.linalg.inv(a)
    except LinAlgError as lae:
        raise ValueError("singular matrix") from lae

def chi2(Npred, Nmeas, covinv, xp=xp):
    '''Return the chi2 value between the expectation value of
    predicted measure Npred and an actual measure Nmeas and INVERSE of
    a covariance matrix invcov.

    Npred and Nmeas may be any mix of scalar shape (nbins,) or batched
    (nbatch, nbins), etc for convinv (nbins,nbins) or
    (nbatch,nbins,nbins).  If any are batched then so is the return.
    If the N's are not batched but the covinv is, the calculation will
    work.  However, this may not be a meaninful thing.
    
    '''
    # see test_tmm.test_ein for sussing this out

    ndims = [Npred.shape, covinv.shape, Nmeas.shape]
    nbs = ndims[0]-1, ndims[1]-2, ndims[2]-1
    eina = 'B'*nbs[0] + 'i'
    einM = 'B'*nbs[1] + 'ij'
    einb = 'B'*nbs[2] + 'j'
    bout = max(nbs)
    einc = 'B'*bout
    ein = f'{eina},{einM},{einb} -> {einc}'
    return xp.einsum(ein, Npred, covinv, Nmeas)

    
