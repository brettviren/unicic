#!/usr/bin/env pytest
import pytest

from itertools import product

import unicic.toy as toy

toyxs = (toy.np,toy.cp,toy.jp)
nbinss = (1,10)
nbatchs = (1,4)


# def pairwise_check_scalar(data):
#     for nbins, bytoy in data.items():
#         # print(f'\nNBINS:{nbins}')
#         toys = list(bytoy)
#         if len(toys) < 2:
#             continue
#         for t1, t2 in zip(toys[:-1], toys[1:]):
#             n1 = bytoy[t1]
#             n2 = bytoy[t2]
#             # print(f'{t1}: {n1}')
#             # print(f'{t2}: {n2}')
#             for v1,v2 in zip(n1,n2):
#                 dv = abs(v1 - v2)
#                 # print(f'\t{v1} - {v2} = {dv}')
#                 assert dv < 1e-6 

def pairwise_check_arrays(data):
    for nn, bytoy in data.items():
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
    

@pytest.fixture(scope="module")
def predict_scalar_results():
    yield {nb:{x:None for x in toyxs} for nb in nbinss}

@pytest.mark.parametrize("toyx,nbins",
                         product(toyxs, nbinss))
def test_predict_scalar(predict_scalar_results, toyx, nbins):
    ms = toyx.xp.linspace(1,10,nbins)
    nparms = 3
    q = toyx.xp.arange(1,nparms+1)
    Npred = toyx.predict_scalar(q, ms)
    assert Npred.shape == (nbins,)
    predict_scalar_results[nbins][toyx] = Npred

def test_predict_scalar_zz(predict_scalar_results):
    pairwise_check_arrays(predict_scalar_results)



@pytest.fixture(scope="module")
def statvar_cnp_scalar_results():
    yield {nb:{x:None for x in toyxs} for nb in nbinss}

@pytest.mark.parametrize("toyx,nbins",
                         product(toyxs, nbinss))
def test_statvar_cnp_scalar(statvar_cnp_scalar_results, toyx, nbins):
    ms = toyx.xp.linspace(1,10,nbins)
    nparms = 3
    q = toyx.xp.arange(1,nparms+1)
    Npred = toyx.predict_scalar(q, ms)
    rng = toyx.Random(42)
    N = toyx.fluctuate(Npred, rng)
    statvar = toyx.statvar_cnp_scalar(Npred, N)
    assert statvar.shape == (nbins, nbins)
    statvar_cnp_scalar_results[nbins][toyx] = statvar

def test_statvar_cnp_scalar_zz(statvar_cnp_scalar_results):
    pairwise_check_arrays(statvar_cnp_scalar_results)





@pytest.fixture(scope="module")
def predict_batched_results():
    yield {nn:{x:None for x in toyxs} for nn in product(nbatchs, nbinss)}

@pytest.mark.parametrize("toyx,nbins,nbatch",
                         product(toyxs, nbinss, nbatchs))
def test_predict_batched(predict_batched_results, toyx, nbins, nbatch):
    ms = toyx.xp.linspace(1,10,nbins)
    nparms = 3
    q = toyx.xp.arange(1,nbatch*nparms+1).reshape(nbatch, nparms)
    N = toyx.predict_batched(q, ms)
    nn = tuple((nbatch,nbins))
    assert N.shape == nn
    predict_batched_results[nn][toyx] = N

def test_predict_batched_zz(predict_batched_results):
    pairwise_check_arrays(predict_batched_results)


@pytest.fixture(scope="module")
def statvar_cnp_batched_results():
    yield {nn:{x:None for x in toyxs} for nn in product(nbatchs, nbinss)}

@pytest.mark.parametrize("toyx,nbins,nbatch",
                         product(toyxs, nbinss, nbatchs))
def test_statvar_cnp_batched(statvar_cnp_scalar_results, toyx, nbins, nbatch):
    ms = toyx.xp.linspace(1,10,nbins)
    nparms = 3
    q = toyx.xp.arange(1,nbatch*nparms+1).reshape(nbatch, nparms)
    Npred = toyx.predict_batched(q, ms)
    rng = toyx.Random(42)
    N = toyx.fluctuate(Npred, rng)
    assert N.shape == Npred.shape
    statvar = toyx.statvar_cnp_scalar(Npred, N)
    assert statvar.shape == (nbatchs, nbins, nbins)
    statvar_cnp_batched_results[nbins][toyx] = statvar

def test_statvar_cnp_batched_zz(statvar_cnp_batched_results):
    pairwise_check_arrays(statvar_cnp_batched_results)
    
