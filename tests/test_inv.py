#!/usr/bin/env pytest
import pytest
import time
from itertools import product

from unicic import low

from unicic.testing import pairwise_check_arrays, timeit

# compare between all toys
lowxs = (low.np,low.cp,low.jp)

epsilon = 1e-6

@pytest.mark.parametrize('lowx',lowxs)
def test_diag(lowx):
    nbins = 100
    rng = lowx.Random(42)
    diag = lowx.uniform(rng, size=nbins)
    m = diag * lowx.xp.eye(nbins)
    minv = lowx.inv(m)
    for d, dinv in zip(diag, lowx.xp.diag(minv)):
        winv = 1.0/d
        diff = abs(winv-dinv)
        if diff > epsilon:
            raise ValueError(f'diag inv {lowx.__name__} diff:{diff} 1/{d} = {winv}, got {dinv}')

@pytest.mark.parametrize('lowx',lowxs)
def test_random(lowx):
    nbins = 100
    rng = lowx.Random(42)
    m = lowx.uniform(rng, size=nbins*nbins).reshape((nbins,nbins))
    minv = lowx.inv(m)

def asotma(n, xp):
    '''Matrix A_n from A Set of Test Matrices

    https://www.ams.org/mcom/1955-09-052/S0025-5718-1955-0074919-9/S0025-5718-1955-0074919-9.pdf

    Sum of column zero of inverse is 1.0.  Other columns sum to 0.0.
    '''
    i = xp.arange(2,n+1,dtype='float64').reshape(-1, 1)
    j = xp.arange(1,n+1,dtype='float64').reshape( 1,-1)
    bot = 1.0/(i + j - 1.0)
    return xp.vstack((xp.ones(n), bot))
    


nbs = 2,10

@pytest.fixture(scope="module")
def astoma_results():
    yield {nb:{t:None for t in lowxs} for nb in nbs}

@pytest.mark.parametrize('lowx,nb',product(lowxs, nbs))
def test_asotma(astoma_results, lowx, nb):
    epsilon = 5e-3
    a = asotma(nb, lowx.xp)
    ainv = lowx.inv(a)
    one = lowx.xp.sum(ainv[:,0])
    if abs(one-1.0) > epsilon:
        raise ValueError(f'asotma column zero sum failed {lowx.__name__} 1.0 != {one}')
    for col in range(1,nb):
        zero = lowx.xp.sum(ainv[:,col])
        if abs(zero) > epsilon:
            raise ValueError(f'asotma column {col} sum failed {lowx.__name__} 0.0 != {zero}')
    astoma_results[nb][lowx] = ainv

def test_astoma_zz(astoma_results):
    epsilon = 5e-5
    pairwise_check_arrays(astoma_results, epsilon)
    

@pytest.mark.parametrize('lowx,nb',product(lowxs, nbs))
def test_roundtrip(lowx, nb):
    epsilon = 5e-5
    a1 = asotma(nb, lowx.xp)
    ainv = lowx.inv(a1)
    a2 = lowx.inv(ainv)
    for x1,x2 in zip(a1.reshape(-1), a2.reshape(-1)):
        d = abs(x1-x2)
        if d > epsilon:
            raise ValueError(f'round trip {lowx.__name__} inaccurate |{x1} - {x2}| = {d} > {epsilon}')



## Double-pump the nbatch=1 case to work around the fact that first
## jax result seems subject to "warmup"
@pytest.mark.parametrize('lowx,nbins,nbatches',product(lowxs, (10,100), (1,1, 10, 100)))
def test_speed(lowx, nbins, nbatches):
    rng = lowx.Random(42)
    size = nbatches*nbins*nbins
    m = lowx.uniform(rng, size=size).reshape((nbatches,nbins,nbins))

    num,mean,sig = timeit(lambda: lowx.inv(m), maxsecs=1, maxn=10000)

    tot = num*mean

    hz = nbatches/mean

    print(f'{lowx.__name__}: {hz:.0f} Hz ({nbins},{nbins}) x {nbatches} * {num} in  mean={mean:.3f} +/- {sig:.4f}, tot={tot:.2f} s')
