#!/usr/bin/env pytest
'''
Test solving the toy for a few points in parameter space.
'''
import numpy
print(numpy.geterr())

import pytest

from itertools import product

from unicic import low
from unicic.toy import Toy
from unicic.util import rebatch
from unicic.testing import pairwise_check_arrays

# the array parameters may be chunked into batches of this size
chunks = (1,10,100,1000)

# true parameters
means = (1.0, 5.0, 20.0)
sigs = (0.5, 5.0, 20.0)
mags = (1.0, 10.0, 100.0)
qtrues = list(product(means, sigs, mags))

# parameter space.  An npbin can be scalar or a triple.  The max/min
# ("mm_") give ranges
npbins = [25, (100,100,100)]
mm_mean = [(0.1,10)]
mm_sig = [(0.1,10)]
mm_mag = [(1.0,100)]
pspaces = list(product(npbins, mm_mean, mm_sig, mm_mag))

# measurement space
nmbins = (10, 100)
maxmeas = (10,)
mspaces = list(product(nmbins, maxmeas))

problems = list(product(qtrues, pspaces, mspaces))

lowxs = (low.np,low.cp,low.jp)

@pytest.fixture(scope="module")
def qbest_results():
    yield {tuple(q):{x:None for x in lowxs} for q in qtrues}

@pytest.mark.parametrize("lowx,problem,chunk", product(lowxs, problems, chunks))
def test_qbest(qbest_results, lowx, problem, chunk):
    'Find qbest for a given problem'

    qtrue,pspace,mspace = problem

    toy = Toy(lowx)

    xp = lowx.xp
    qtrue = xp.array(qtrue)

    npbin,*pmm = pspace
    pmm = xp.array(pmm)         # (3,2)
    pspace = lowx.linspace(pmm[:,0], pmm[:,1], npbin)
    ## fixme: ^^^^ rename linspace and produce flat (N,3)
    nmbin,*mmax = mspace
    mspace = lowx.linspace(None, mmax, nmbin)

    Npred = toy.predict(qtrue, mspace)

    rng = lowx.Random(42)
    Nmeas = lowx.fluctuate(Npred, rng)

    pspace_batch, pspace_remain = rebatch(pspace, chunk)
    bpspaces = [p for p in pspace_batch]
    if pspace_remain.shape[0]:
        bpspaces.append(pspace_remain)

    def chi2(q):
        print(f'chi2: q:{q.shape}')
        npred = toy.predict(q, mspace)
        cov = toy.covariance(npred, Nmeas)
        try:
            covinv = lowx.inv(cov)
        except ValueError as err:
            print(f'covariance failure over {q.shape}: "{err}"')
            return xp.array([xp.nan]*q.shape[0])

        print('chi2:',q.shape, npred.shape, covinv.shape)
        return toy.chi2(npred, Nmeas, covinv)

    def append(l, a):
        l.append(a)
        return l

    chi2s = xp.hstack(lowx.map_reduce(chi2, append, bpspaces, initial=list()))

    maxchi2 = 9e9
    bad = xp.zeros_like(chi2s) + maxchi2
    nans = lowx.xp.isnan(chi2s)
    chi2s = xp.where(nans, bad, chi2s)

    indbest = xp.argmin(chi2s)
    qbest = pspace[indbest]

    print(indbest, qbest, qtrue)
