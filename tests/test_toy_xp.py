#!/usr/bin/env pytest
import pytest

from itertools import product

from unicic.testing import pairwise_check_arrays

from unicic import low
from unicic.toy import Toy

# compare between all toys
lowxs = (low.np,low.cp,low.jp)

# the number of measurement bins to try
nbinss = (1,10)

# the number of batches to try
nbatchs = (1,4)

def predict_scalar(lowx, nbins):
    ms = lowx.linspace(1,10,nbins)
    nparms = 3
    q = lowx.xp.arange(1,nparms+1)
    toy = Toy(lowx)
    Npred = toy.predict(q, ms)
    return Npred

@pytest.fixture(scope="module")
def predict_scalar_results():
    yield {nb:{x:None for x in lowxs} for nb in nbinss}

@pytest.mark.parametrize("lowx,nbins",
                         product(lowxs, nbinss))
def test_predict_scalar(predict_scalar_results, lowx, nbins):
    Npred = predict_scalar(lowx, nbins)
    if Npred.shape != (nbins,):
        raise ValueError(f'bad shape for Npred api={lowx.__name__}, nbins={nbins} != Npred:{Npred.shape}')
    predict_scalar_results[nbins][lowx] = Npred

def test_predict_scalar_zz(predict_scalar_results):
    pairwise_check_arrays(predict_scalar_results)


def do_covariance_scalar(lowx, nbins):
    Npred = predict_scalar(lowx, nbins)
    rng = lowx.Random(42)
    Ndata = lowx.fluctuate(Npred, rng)
    toy = Toy(lowx)
    cov = toy.covariance(Npred, Ndata)
    return cov, Ndata, Npred
    

@pytest.mark.parametrize("lowx,nbins",
                         product(lowxs, nbinss))
def test_covariance_cnp_scalar(lowx, nbins):
    cov, _, _ = do_covariance_scalar(lowx, nbins)
    nn = tuple((nbins,nbins))
    if cov.shape != nn:
        raise ValueError(f'bad shape for covariance api={lowx.__name__} {cov.shape} != {nn}')


### batched


def do_predict_batched(lowx, nbins, nbatch):
    ms = lowx.linspace(1,10,nbins)
    nparms = 3
    q = lowx.xp.arange(1,nbatch*nparms+1).reshape(nbatch, nparms)
    toy = Toy(lowx)
    Npred = toy.predict(q, ms)
    return Npred

@pytest.fixture(scope="module")
def predict_batched_results():
    yield {nn:{x:None for x in lowxs} for nn in product(nbatchs, nbinss)}

@pytest.mark.parametrize("lowx,nbins,nbatch",
                         product(lowxs, nbinss, nbatchs))
def test_predict_batched(predict_batched_results, lowx, nbins, nbatch):
    Npred = do_predict_batched(lowx, nbins, nbatch)

    nn = tuple((nbatch,nbins))
    if Npred.shape != nn:
        raise ValueError(f'bad shape for Npred {Npred.shape} != {nn}')
    predict_batched_results[nn][lowx] = Npred

def test_predict_batched_zz(predict_batched_results):
    pairwise_check_arrays(predict_batched_results)


def do_fluctuate(lowx, nbins, nbatch):
    Npred = do_predict_batched(lowx, nbins, nbatch)
    rng = lowx.Random(42)
    Ndata = lowx.fluctuate(Npred, rng)
    return Ndata, Npred

@pytest.mark.parametrize("lowx,nbins,nbatch",
                         product(lowxs, nbinss, nbatchs))
def test_covariance_cnp_batched(lowx, nbins, nbatch):
    Ndata, Npred = do_fluctuate(lowx, nbins, nbatch)
    if Ndata.shape != Npred.shape:
        raise ValueError(f'bad shape nbins:{nbins} nbatch:{nbatch} api={lowx.__name__} Ndata:{Ndata.shape} != Npred:{Npred.shape}')

    toy = Toy(lowx)
    cov = toy.covariance(Npred, Ndata)

    if cov.shape != (nbatch, nbins, nbins):
        raise ValueError(f'bad shape nbins:{nbins} nbatch:{nbatch} api={lowx.__name__} cov:{cov.shape} != {(nbatch, nbins, nbins)}')

