#!/usr/bin/env python3
'''
The "toy" problem for testing, implemented for jax
'''

import unicic.low.jp as low

xp = low.xp

# Actual code is same as np's, parameterized by the arry module xp.
from . import np as _np

def predict(q, ms, xp=xp):
    return _np.predict(q, ms, xp)

def statvar_cnp(Npred, Ndata, xp=xp):
    return _np.statvar_cnp(Npred, Ndata, xp=xp)

chi2 = _np.chi2

