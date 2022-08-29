#!/usr/bin/env python3
'''
Utility functions for writing tests for unicic.  
'''

import time
import math

def pairwise_check_arrays(data, epsilon=1e-6):
    '''Assure consistent results between parts of data.

    Data is a nested dict indexed by [test][toy]=array

    '''
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
                dv = abs(v1 - v2)/(abs(v1)+abs(v2))
                if dv < epsilon:
                    continue
                print(f'\nindex:{nn}')
                print(f'\n{t1.__name__}:{v1}')
                print(f'\n{t2.__name__}:{v2}')
                print(f'\ndiff:{dv}')
                raise ValueError(f'large difference {dv} between {t1.__name__} and {t2.__name__} for {nn}: {v1} != {v2}')
    


def timeit(thunk, maxsecs=10, maxn=10):
    '''Time a thunk.

    Thunk is called up to maxn times but no more than maxsecs seconds.

    '''

    laps = [time.perf_counter()]
    for n in range(maxn):
        thunk()
        laps.append(time.perf_counter())
        cur = laps[-1] - laps[0]
        est = cur / (n+1)
        if (cur + est > maxsecs):
            break
    dt = [a-b for a,b in zip(laps[1:], laps[:-1])]
    dt2 = [(a-b)**2 for a,b in zip(laps[1:], laps[:-1])]

    N = len(dt)
    mean = sum(dt)/N
    sigma = math.sqrt(sum(dt2)/N - mean**2)
    return N,mean,sigma
            
