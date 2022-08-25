#!/usr/bin/env python3
'''A toy exercising Feldman-Cousins Unified Approach to constructing
confidence regions .

Notation:

- p, q ::  point in space of parameters input to model.
- N :: point in space of measurements output by model.
- Npred :: central expectation of measurement assuming a q.
- Nfluc :: fluctuation of expectation.

Batched versions of variables may be provided with the batched axis=0.

'''

import cupy as cp
import numpy as np

from cupy_backends.cuda.libs.cusolver import CUSOLVERError
from numpy.linalg import LinAlgError

from functools import partial, lru_cache # , cache, cached_property

# Default measurement space
default_ms = np.linspace(0,10,100,False)

npbins = 25

# Default parameter space
default_ps = (
    np.linspace(1., 10, npbins, False), # mu
    np.linspace(1., 10, npbins, False), # sig
    np.linspace(10., 110, npbins, False)  # mag
)

class SGD:
    def __init__(self, learning_rate=1.0):
        self._lr = learning_rate
    def __call__(self, gradients, state):
        return -self._lr * gradients, state

class Toy:
    def __init__(self, xp=np, ms = default_ms, pses = default_ps):
        ''' Create toy calculation with measurement linspace ms and
        parameter linspaces *ps '''
        self.ms = xp.array(ms)
        self.ps = tuple([xp.array(ps) for ps in pses])
        self.xp = xp
        # make and reuse to save a few %
        self.I = self.xp.eye(ms.size)
        

    #@cached_property
    @property
    @lru_cache()
    def ps_mesh(self):
        'A meshgrid spanning parameter space'
        return self.xp.meshgrid(*self.ps)

    #@cached_property
    @property
    @lru_cache()
    def ps_flat(self):
        'Parameter space points shaped (Npoints, Ndims)'
        return self.xp.stack([one.reshape(-1) for one in self.ps_mesh]).T
    
    @property
    def qrand(self):
        'A random point in parameter space'
        q = self.ps_flat
        ind = self.xp.random.randint(0, q.shape[0])
        return q[ind]

    # fixme: @cache or @lru_cache?
    def predict(self, q):
        '''Predict expectated central value of measurements of q.

        if q is (3,) array, return scalar array shaped ()

        if q is (n,3) array (batched), return array shaped (n,nbins)

        '''
        squeeze = False
        if len(q.shape) == 1:
            squeeze = True
            q = q.reshape(1,3)

        ms = self.xp.expand_dims(self.ms, axis=0)
        mu,sig,mag = self.xp.expand_dims(q.T, axis=2)

        # print(f'predict: q:{q.shape} mu:{mu.shape}')
        gnorm = mag / (self.xp.sqrt(2*self.xp.pi))

        d = self.ms - mu
        ret = gnorm * self.xp.exp(-0.5*((d/sig)**2))
        if squeeze:
            ret = self.xp.squeeze(ret)
        # print(f'predict: ret:{ret.shape}')
        return ret

    def statv_cnp(self, N, q):
        '''
        Statistical covariance matrix following combined neyman-pearson.

        q may be scalar shaped (3,) or batched shaped (n,3).

        N shape is (nbins,) and must not be batched.

        Scalar return (nbins,nbins) or batched is (n,nbins,nbins).
        '''
        # fixme: handled batched N, batched q or both

        Npred = self.predict(q) # may be batched (n, nbins)
        num = 3 * N * Npred
        den = 2*N + Npred

        # guard against zero and divide-by-zero
        good_num = num > 0
        num = self.xp.where(good_num, num, 0.5*Npred)
        den = self.xp.where(good_num, den, 1.0)

        good_den = den > 0
        num = self.xp.where(good_den, num, 0.5*Npred)
        den = self.xp.where(good_den, den, 1.0)

        diag = num/den

        # diag = 3.0/(1.0/N + 2.0/Npred)
        # diag = self.xp.where(diag > 0, diag, N)
        # diag = self.xp.where(diag > 0, diag, 0.5*Npred)
        # see appendix a of CNP paper for the 1/2.

        I = self.I
        if len(q.shape) > 1:    # batched
            I = self.xp.expand_dims(I, axis=0) # add batch
            diag = self.xp.expand_dims(diag, axis=1) 
            
        c = diag * I
        return c

    def covariance(self, N, q):
        '''
        Full covariance
        '''
        # for now, just stats
        return self.statv_cnp(N, q);
    
    def fluctuate(self, q):
        '''
        Return fluctuated measure of expectation
        '''
        npred = self.predict(q)
        nfluc = self.xp.random.poisson(npred)
        return nfluc

    def chi2(self, N, q):
        '''
        Return the chi2 value for the measurement and a prediction at q.

        q may be batched.

        '''
        if len(q.shape) == 1:    # not batched
            q = self.xp.expand_dims(q, axis=0)

        npreds = self.predict(q)  # (n, nbins)
        cs = self.covariance(N,q) # (n, nbins, nbins)

        try:
            cinv = self.xp.linalg.inv(cs) # (n, nbins, nbins)
        except LinAlgError:
            for c in cs:
                print(self.xp.diag(c))
            raise
        dN = self.xp.expand_dims(N - npreds, axis=2)
        dNT = self.xp.transpose(dN, axes=(0,2,1))
        ret = (dNT @ cinv @ dN).squeeze()
        return ret


    def most_likely_gs(self, N, chunk_size=1000):
        '''Return best fit parameter through grid search.

        Best fit parameter minimizes chi2 between given measure N and
        the prediction at the parameter.

        An exhaustive grid search is used.

        The chunk_size sets number of parameter values tested in one
        call to a vmap'ed chi2.  It may be increased or reduced to
        match available GPU memory.

        '''

        # (n,3)
        qs = self.ps_flat

        ## no vmap in cupy
        # @vmap
        def Nchi2(qs):
            print(f'Nchi2 qs:{qs.shape}')
            res = list()
            for q in qs:
                res.append(self.chi2(N, q))
            return self.xp.stack(res)

        ndevs = 1
        npoints = int(qs.shape[0])
        nperdev = npoints // ndevs
        nchunks = nperdev // chunk_size
        if nchunks == 0:
            err = f'failed to find solution for parallel distribution: chunk size {chunk_size} is too large or ndevs {ndevs} is too large for {npoints} points'
            print(err)
            raise ValueError(err)
        npar = ndevs * nchunks * chunk_size
        print(f'ndevs={ndevs} nperdev={nperdev} nchunks={nchunks} npar={npar} npoints={npoints}')

        qspar = qs[:npar,:].reshape(ndevs, nchunks, chunk_size, -1)
        print(f'qspar:{qspar.shape}')

        def _bydev(qschunks):
            print(f'qschunks:{qschunks.shape}')
            many = list()
            for qschunk in qschunks:
                one = Nchi2(qschunk)
                many.append(one)
            ret = self.xp.hstack(many)
            print(f'bydev:{ret.shape}')
            return ret
        ## no pmap in cupy
        # bydev = pmap(bydev)
        def bydev(qspar):
            res = list()
            for qschunks in qspar:
                res.append(_bydev(qschunks))
            return self.xp.stack(res)

        print('calling bydev')
        parts = bydev(qspar)
        parts = parts.reshape(-1)
        print(f'parts:{parts.shape}')
        
        leftovers = Nchi2(qs[npar:,:])
        chi2s = self.xp.hstack((parts, leftovers))
        print(f'chi2s:{chi2s.shape}')
        
        nans = self.xp.isnan(chi2s)
        if self.xp.any(nans):
            nums = self.xp.invert(nans)
            maxchi2 = self.xp.max(chi2s[nums])
            print(f'NaNs in chi2, replacing with max chi2: {maxchi2}')
            bad = self.xp.zeros_like(chi2s) + maxchi2
            chi2s = self.xp.where(nans, bad, chi2s)
        indbest = self.xp.argmin(chi2s)
        chi2best = chi2s[indbest]
        qbest = qs[indbest,:]
        print (qbest, chi2best)
        return qbest, chi2best, chi2s


    # def most_likely_opt(self, N, q, opt_fn, opt_state=None, steps=100):
    #     '''Return parameter with the prediction that minimizes chi2 with
    #     the given measure N.  The parameter point q gives a starting point'''

    #     Nchi2 = jit(vmap(partial(self.chi2, N)))

    #     losses = []
    #     qpoints = []
    #     for _ in range(steps):
    #         loss, grads = value_and_grad(Nchi2)(q)
    #         updates, opt_state = opt_fn(grads, opt_state)
    #         q += updates
    #         qpoints.append(q)
    #         losses.append(loss)
    #     return self.xp.stack(losses), self.xp.stack(qpoints), q, opt_state
    

    # def delta_chi2(self, N, p):
    #     '''Return the difference of two chi2 values.  First is the chi2
    #     value for the measurement N vs the prediction at p.  Second is the
    #     chi2 value for the measurement N and the prediction qbest which
    #     maximizes the likelihood to have produced N.
    #     '''
    #     chi2null = self.chi2(N, p)
    #     qbest = most_likely_gs(N) # ...fixme...
    #     chi2best = self.chi2(N, qbest)
    #     return chi2null - chi2best


###


def check_qs(t, num):

    q = t.xp.array([t.qrand for n in range(num)])
    print(q.shape)
    print(q)

    Npred = t.predict(q)
    Ns = t.fluctuate(q)
    c = t.xp.array([t.covariance(N, q) for N in Ns])
    print (f'shapes: {q.shape} {Npred.shape} {Ns.shape} c:{c.shape}')

    chi2s = t.xp.array([t.chi2(N, q) for N in Ns])
    print(f'chi2s: {chi2s}')

    # print('chi2(fluctuated, q) =', t.chi2(N, q))
    # print('chi2(predicted, q)  =', t.chi2(Npred, q))

def test_some_things(xp=np, chunk_size=1000):
    t = Toy(xp)

    check_qs(t, 2)

    q = t.qrand
    N = t.fluctuate(q)

    def my_func():
        qbest, chi2best, chi2s = t.most_likely_gs(N, chunk_size)
        print(f'q=\n{q}\nqbest=\n{qbest}\ndiff=\n{q-qbest}\nchi2best={chi2best}')

    # from cupyx.profiler import benchmark
    #print(benchmark(my_func, (), n_repeat=1, n_warmup=0))

    import time
    start = time.perf_counter()
    my_func()
    stop = time.perf_counter()
    dt = stop-start
    print(f'elapsed={dt:.3f} s rate={len(t.ps_flat)/dt:.3f} Hz')

    # qbest, chi2best, chi2s = t.most_likely_gs(N, chunk_size)
    # print(f'q=\n{q}\nqbest=\n{qbest}\ndiff=\n{q-qbest}\nchi2best={chi2best}')
    return locals()

if "__main__" == __name__:
    import sys
    xp = np
    chunk_size = 1000
    for arg in sys.argv[1:]:
        if arg in ("cupy","cp"):
            xp = cp
            continue
        try:
            chunk_size = int(arg)
        except ValueError:
            continue
            
    test_some_things(xp, chunk_size)

# eg:
# python toy.py
# or:
# XLA_FLAGS='--xla_force_host_platform_device_count=10' python toy.py
