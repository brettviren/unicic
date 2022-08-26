#!/usr/bin/env pytest
import pytest
import time
from itertools import product

import unicic.toy as toy
from unicic.testing import pairwise_check_arrays

# compare between all toys
toyxs = (toy.np,toy.cp,toy.jp)


epsilon = 1e-6

@pytest.mark.parametrize('toyx',toyxs)
def test_diag(toyx):
    nbins = 100
    rng = toyx.Random(42)
    diag = toyx.uniform(rng, size=nbins)
    m = diag * toyx.xp.eye(nbins)
    minv = toyx.inv(m)
    for d, dinv in zip(diag, toyx.xp.diag(minv)):
        winv = 1.0/d
        diff = abs(winv-dinv)
        if diff > epsilon:
            raise ValueError(f'diag inv {toyx.__name__} diff:{diff} 1/{d} = {winv}, got {dinv}')

@pytest.mark.parametrize('toyx',toyxs)
def test_random(toyx):
    nbins = 100
    rng = toyx.Random(42)
    m = toyx.uniform(rng, size=nbins*nbins).reshape((nbins,nbins))
    minv = toyx.inv(m)

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
    yield {nb:{t:None for t in toyxs} for nb in nbs}

@pytest.mark.parametrize('toyx,nb',product(toyxs, nbs))
def test_asotma(astoma_results, toyx, nb):
    epsilon = 5e-3
    a = asotma(nb, toyx.xp)
    ainv = toyx.inv(a)
    one = toyx.xp.sum(ainv[:,0])
    if abs(one-1.0) > epsilon:
        raise ValueError(f'asotma column zero sum failed {toyx.__name__} 1.0 != {one}')
    for col in range(1,nb):
        zero = toyx.xp.sum(ainv[:,col])
        if abs(zero) > epsilon:
            raise ValueError(f'asotma column {col} sum failed {toyx.__name__} 0.0 != {zero}')
    astoma_results[nb][toyx] = ainv

def test_astoma_zz(astoma_results):
    epsilon = 5e-5
    pairwise_check_arrays(astoma_results, epsilon)
    

@pytest.mark.parametrize('toyx,nb',product(toyxs, nbs))
def test_roundtrip(toyx, nb):
    epsilon = 5e-5
    a1 = asotma(nb, toyx.xp)
    ainv = toyx.inv(a1)
    a2 = toyx.inv(ainv)
    for x1,x2 in zip(a1.reshape(-1), a2.reshape(-1)):
        d = abs(x1-x2)
        if d > epsilon:
            raise ValueError(f'round trip {toyx.__name__} inaccurate |{x1} - {x2}| = {d} > {epsilon}')


## this fails on jax due to memory outage.  someone please buy me a big GPU!!
@pytest.mark.parametrize('toyx,nbins,nbatches',product(toyxs[:-1], (10,100), (1, 10, 100, 1000)))
def test_speed(toyx, nbins, nbatches):
    rng = toyx.Random(42)
    size = nbatches*nbins*nbins
    m = toyx.uniform(rng, size=size).reshape((nbatches,nbins,nbins))
    rounds = 1 + (100*100*100) // size
    # print(f'{rounds} rounds of {nbatches} x ({nbins},{nbins})')
    t1 = time.perf_counter()
    for _ in range(rounds):
        minv = toyx.inv(m)
    t2 = time.perf_counter()
    dt = t2-t1
    each = dt / (rounds*nbatches)
    print(f'{toyx.__name__}: {rounds} rounds of {nbatches} x ({nbins},{nbins}) in {dt:.3f} s, {1.0/each:.1f} Hz')
