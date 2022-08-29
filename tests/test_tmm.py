#!/usr/bin/env pytest
import pytest

from itertools import product

import unicic.toy as toy

# compare between all toys
toyxs = (toy.np, toy.cp, toy.jp)


# In chi2 we multiply three: a @ M @ b
#
# Want to write a function to do this which allows a or M or b to be
# batched.  All combinations of batched/not-batched are allowed with
# the exception that if neither a nor b are batched then M is not
# batched.

def make_shapes(n=3, nb=5):
    'List of shapes for n bins and nb batches'
    return [
        [(n,),    (    n,n), (n,)],
        [(n,),    (    n,n), (nb,n)],
        [(nb, n), (    n,n), (n,)],
        [(nb, n), (    n,n), (nb, n)],
        [(n,),    (nb, n,n), (nb,n)],
        [(nb, n), (nb, n,n), (n,)],
        [(nb, n), (nb, n,n), (nb, n)],
        [(n,),    (nb, n,n), (n,)],
    ]

# All n/c/j have einsum() but perhaps constructing the needed strings
# is expensive?

@pytest.mark.parametrize('toyx,shapes',
                         product(toyxs, make_shapes()))
def test_ein(toyx, shapes):
    rng = toyx.low.Random(42) 

    arrs = list()
    for shape in shapes:
        size = int(toyx.xp.product(toyx.xp.array(shape, dtype=int)))
        arr = toyx.low.uniform(rng, size = size)
        arr = arr.reshape(shape)
        arrs.append(arr)

    a,M,b = arrs
    # print(f'\n{a.shape} @ {M.shape} @ {b.shape}')
    ndims = [len(s) for s in shapes]

    nbs = ndims[0]-1, ndims[1]-2, ndims[2]-1

    eina = 'B'*nbs[0] + 'i'
    einM = 'B'*nbs[1] + 'ij'
    einb = 'B'*nbs[2] + 'j'

    bout = max(nbs)
    einc = 'B'*bout

    ein = f'{eina},{einM},{einb} -> {einc}'
    # print(ein)
    c = toyx.xp.einsum(ein, a, M, b)

    # print(f'\n{a.shape} @ {M.shape} @ {b.shape} = {c.shape}')

    if len(c.shape) != bout:
        raise ValueError(f'bad shape: {toyx.__name__} {shapes}: {c}')


