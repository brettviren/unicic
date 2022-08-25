#!/usr/bin/env python3
'''
The "toy" problem for testing, implemented for numpy/cupy
'''

import numpy
xp = numpy

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
