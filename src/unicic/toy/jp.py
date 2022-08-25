#!/usr/bin/env python3
'''
The "toy" problem for testing, implemented for jax
'''

import jax.numpy as xp

from jax import vmap

from . import np

def predict_scalar(q, ms, xp=xp):
    return np.predict_scalar(q, ms, xp)

predict_batched = vmap(predict_scalar, in_axes=(0,None), out_axes=0)

