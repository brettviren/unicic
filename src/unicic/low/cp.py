#!/usr/bin/env python3
'''
The low level unicic api for cupy
'''

import time

import cupy
xp = cupy

from functools import reduce

# some things can share code with the np implementation
from . import np as _np

Random = cupy.random.default_rng
def uniform(rng, low=0.0, high=1.0, size=None):
    r = rng.random(size)
    return r * (high - low) + low

def fluctuate(Npred, rng, xp=xp):
    return rng.poisson(Npred)

def inv(a, xp=xp):
    '''
    Return the multiplicative inverse of a.

    Raises ValueError if singular
    '''
    # fixme: this error catching is not correct for cupy
    try:
        return xp.linalg.inv(a)
    except LinAlgError as lae:
        raise ValueError("singular matrix") from lae

def linspace(start, stop, num, endpoint=True, xp=xp):
    return _np.linspace(start, stop, num, endpoint=endpoint, xp=xp)


def map_reduce(mfunc, rfunc, iterable, *, initial=None, nproc=1):
    '''Map mfunc on each in iterable and reduce that with rfunc.

    If initial is given it is provided first to rfunc.

    If nproc > 1 multiple CUDA streams will be used on the current device.
    '''
    start_time = time.time()

    if nproc > 1:
        #print(f'warning: map_reduce for cupy is currently only serial')
        nproc = 1

    if nproc == 1:
        res = _np.map_reduce(mfunc, rfunc, iterable, initial=initial, nproc=1)
        dt = time.time() - start_time
        return res

    # follows https://github.com/cupy/cupy/blob/master/examples/stream/map_reduce.py
    ## it's unstable
    dev = cupy.cuda.Device()
    mempool = cupy.cuda.MemoryPool()
    cupy.cuda.set_allocator(mempool.malloc)

    rstream = cupy.cuda.stream.Stream(non_blocking=True)
    mstreams = [cupy.cuda.stream.Stream(non_blocking=True) for n in range(nproc)]

    mapped = list()
    for ind, thing in enumerate(iterable):
        stream = mstreams[ind%nproc]
        with stream:
            one = mfunc(thing)
            mapped.append(one)

    stop_events = list()
    for stream in mstreams:
        se = stream.record()
        stop_events.append(se)

    for se in stop_events:
        rstream.wait_event(se)

    with rstream:
        if initial is None:
            res = reduce(rfunc, mapped)
        res = reduce(rfunc, mapped, initial)
    # if initial is None:
    #     res = reduce(rfunc, mapped)
    # res = reduce(rfunc, mapped, initial)

    dt = time.time() - start_time
    pre=mempool.total_bytes()

    dev.synchronize()
    for stream in mstreams:
        mempool.free_all_blocks(stream=stream)

    post=mempool.total_bytes()

    print(f'nproc={nproc} dt={dt} mem: {pre} -> {post} diff: {pre-post}')
    return res
