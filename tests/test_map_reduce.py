#!/usr/bin/env pytest
import pytest

from itertools import product

from unicic import low
from unicic.testing import pairwise_check_arrays, timeit

# Import low-level API
lowxs = (low.np, low.cp, low.jp)


nbatches = (1,10,100,1000)
nsizes = (10,100)
#nprocs = (1,2,5,10,100)
nprocs = (1,)
# 
# nbatches = (100,)
# nsizes = (100,)
# nprocs = (1,2,10)

def square_sum(a):
    return (a*a).sum()

def add(a,b):
    return a+b

@pytest.mark.parametrize('lowx,nbatch,nsize,nproc',
                         product(lowxs, nbatches, nsizes, nprocs))

def test_map_reduce(lowx, nbatch, nsize, nproc):
    xp = lowx.xp
    arr = xp.arange(nbatch*nsize*nsize).reshape(nbatch,nsize,nsize)

    def doit():
        lowx.map_reduce(square_sum, add, arr, initial=0, nproc=nproc)        

    num,mean,sig = timeit(doit, maxsecs=1, maxn=100000)
    hz = 0.001*nbatch/mean

    print(f'{lowx.__name__}\t{hz:.1f} kHz\t{mean*1000:.3f}ms +/- {sig*1000:.5f}ms\t{nbatch}*({nsize},{nsize})\t({nproc}) {num}')



    
