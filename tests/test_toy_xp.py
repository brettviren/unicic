#!/usr/bin/env pytest
import pytest

from itertools import product

import unicic.toy as toy

toyxs = (toy.np,toy.cp,toy.jp)
nbinss = (1,10)
nbatchs = (1,4)

@pytest.fixture(scope="module")
def scalars():
    yield {nb:{x:None for x in toyxs} for nb in nbinss}

@pytest.mark.parametrize("toyx,nbins",
                         product(toyxs, nbinss))
def test_predict_scalar(scalars, toyx, nbins):
    ms = toyx.xp.linspace(1,10,nbins)
    nparms = 3
    q = toyx.xp.arange(1,nparms+1)
    N = toyx.predict_scalar(q, ms)
    assert N.shape == (nbins,)
    scalars[nbins][toyx] = N

def test_predict_scalar_zz(scalars):
    for nbins, bytoy in scalars.items():
        # print(f'\nNBINS:{nbins}')
        toys = list(bytoy)
        if len(toys) < 2:
            continue
        for t1, t2 in zip(toys[:-1], toys[1:]):
            n1 = bytoy[t1]
            n2 = bytoy[t2]
            # print(f'{t1}: {n1}')
            # print(f'{t2}: {n2}')
            for v1,v2 in zip(n1,n2):
                dv = abs(v1 - v2)
                # print(f'\t{v1} - {v2} = {dv}')
                assert dv < 1e-6 

@pytest.fixture(scope="module")
def batches():
    yield {nn:{x:None for x in toyxs} for nn in product(nbatchs, nbinss)}

@pytest.mark.parametrize("toyx,nbins,nbatch",
                         product(toyxs, nbinss, nbatchs))
def test_predict_batched(batches, toyx, nbins, nbatch):
    ms = toyx.xp.linspace(1,10,nbins)
    nparms = 3
    q = toyx.xp.arange(1,nbatch*nparms+1).reshape(nbatch, nparms)
    N = toyx.predict_batched(q, ms)
    nn = tuple((nbatch,nbins))
    assert N.shape == nn
    batches[nn][toyx] = N

def test_predict_batched_zz(batches):
    for nn, bytoy in batches.items():
        # print(f'\nNN:{nn}')
        toys = list(bytoy)
        if len(toys) < 2:
            continue
        for t1, t2 in zip(toys[:-1], toys[1:]):
            n1 = bytoy[t1]
            n2 = bytoy[t2]
            # print(f'{t1}: {n1}')
            # print(f'{t2}: {n2}')
            for v1,v2 in zip(n1.reshape(-1),n2.reshape(-1)):
                dv = abs(v1 - v2)
                # print(f'\t{v1} - {v2} = {dv}')
                assert dv < 1e-6 
    
