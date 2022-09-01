#!/usr/bin/env pytest
'''
Test toy covariance correctness
'''

import pytest
from itertools import product

from unicic import low
from unicic.toy import Toy


lowxs = (low.np,low.cp,low.jp)
qtrues = product((1.0,5.0, 11.0),
                 (0.5, 5.0, 20.0),
                 (1.0, 10.0, 100.0))                 


@pytest.mark.parametrize("lowx,qtrue", product(lowxs, qtrues))
def test_toy_cov_nonzero(lowx,qtrue):
    toy = Toy(lowx)
    xp = lowx.xp
    mspace = lowx.linspace(0, 10, 100)
    Npred = toy.predict(qtrue, mspace)
    if not xp.all(Npred > 0):
        print (f'\nNpred:\n{Npred}\nqtrue:\n{qtrue}')
        raise ValueError('Predicted measure has zeros')

    rng = lowx.Random(42)
    Nmeas = lowx.fluctuate(Npred, rng)

    cov = toy.covariance(Npred, Nmeas)
    covd = xp.diag(cov)
    assert xp.all(covd > 0)

    covinv = lowx.inv(cov)
    

    
