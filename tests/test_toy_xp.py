#!/usr/bin/env pytest
import pytest

from itertools import product

import unicic.toy as toy

# compare between all toys
toyxs = (toy.np,toy.cp,toy.jp)

# the number of measurement bins to try
nbinss = (1,10)

# the number of batches to try
nbatchs = (1,4)

# numeric precision allowed
epsilon = 1e-6

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
                if dv < epsilon:
                    continue
                print(f'\nindex:{nn}')
                print(f'\n{t1.__name__}:{v1}')
                print(f'\n{t2.__name__}:{v2}')
                print(f'\ndiff:{dv}')
                raise ValueError(f'large difference between {t1.__name__} and {t2.__name__} for {nn}')
    

def get_msq(toyx, nbins):
    ms = toyx.xp.linspace(1,10,nbins)
    nparms = 3
    q = toyx.xp.arange(1,nparms+1)
    return ms,q

def predict_scalar(toyx, nbins):
    ms,q = get_msq(toyx, nbins)
    Npred = toyx.predict_scalar(q, ms)
    return Npred

@pytest.fixture(scope="module")
def predict_scalar_results():
    yield {nb:{x:None for x in toyxs} for nb in nbinss}

@pytest.mark.parametrize("toyx,nbins",
                         product(toyxs, nbinss))
def test_predict_scalar(predict_scalar_results, toyx, nbins):
    Npred = predict_scalar(toyx, nbins)
    if Npred.shape != (nbins,):
        raise ValueError(f'bad shape for Npred toy={toyx.__name__}, nbins={nbins} != Npred:{Npred.shape}')
    predict_scalar_results[nbins][toyx] = Npred

def test_predict_scalar_zz(predict_scalar_results):
    pairwise_check_arrays(predict_scalar_results)


def do_statvar_scalar(toyx, nbins):
    Npred = predict_scalar(toyx, nbins)
    rng = toyx.Random(42)
    Ndata = toyx.fluctuate(Npred, rng)
    return toyx.statvar_cnp(Npred, Ndata), Ndata, Npred
    

@pytest.mark.parametrize("toyx,nbins",
                         product(toyxs, nbinss))
def test_statvar_cnp_scalar(toyx, nbins):
    statvar, _, _ = do_statvar_scalar(toyx, nbins)
    nn = tuple((nbins,nbins))
    if statvar.shape != nn:
        raise ValueError(f'bad shape for statvar toy={toyx.__name__} {statvar.shape} != {nn}')


### batched


def do_predict_batched(toyx, nbins, nbatch):
    ms = toyx.xp.linspace(1,10,nbins)
    nparms = 3
    q = toyx.xp.arange(1,nbatch*nparms+1).reshape(nbatch, nparms)
    Npred = toyx.predict_batched(q, ms)
    return Npred

@pytest.fixture(scope="module")
def predict_batched_results():
    yield {nn:{x:None for x in toyxs} for nn in product(nbatchs, nbinss)}

@pytest.mark.parametrize("toyx,nbins,nbatch",
                         product(toyxs, nbinss, nbatchs))
def test_predict_batched(predict_batched_results, toyx, nbins, nbatch):
    Npred = do_predict_batched(toyx, nbins, nbatch)

    nn = tuple((nbatch,nbins))
    if Npred.shape != nn:
        raise ValueError(f'bad shape for Npred {Npred.shape} != {nn}')
    predict_batched_results[nn][toyx] = Npred

def test_predict_batched_zz(predict_batched_results):
    pairwise_check_arrays(predict_batched_results)


def do_fluctuate(toyx, nbins, nbatch):
    Npred = do_predict_batched(toyx, nbins, nbatch)
    rng = toyx.Random(42)
    Ndata = toyx.fluctuate(Npred, rng)
    return Ndata, Npred

@pytest.mark.parametrize("toyx,nbins,nbatch",
                         product(toyxs, nbinss, nbatchs))
def test_statvar_cnp_batched(toyx, nbins, nbatch):
    Ndata, Npred = do_fluctuate(toyx, nbins, nbatch)
    if Ndata.shape != Npred.shape:
        raise ValueError(f'bad shape nbins:{nbins} nbatch:{nbatch} toy={toyx.__name__} Ndata:{Ndata.shape} != Npred:{Npred.shape}')
    statvar = toyx.statvar_cnp(Npred, Ndata)
    if statvar.shape != (nbatch, nbins, nbins):
        raise ValueError(f'bad shape nbins:{nbins} nbatch:{nbatch} toy={toyx.__name__} statvar:{statvar.shape} != {(nbatch, nbins, nbins)}')

