#!/usr/bin/env python3
'''

A toy exercising Feldman-Cousins Unified Approach to constructing
confidence regions .

Notation:

- p, q ::  point in space of parameters input to model.
- N :: point in space of measurements output by model.
- Npred :: central expectation of measurement assuming a q.
- Nfluc :: fluctuation of expectation.

'''
import jax.config

import sys
if "cpu" in sys.argv:           # must do this early?
    jax.config.update('jax_platform_name', "cpu")
# also set XLA_FLAGS='--xla_force_host_platform_device_count=40

import jax.numpy as jnp
import numpy as np
from jax import random, pmap, vmap, jit, value_and_grad
from functools import partial, cache, cached_property

# Default measurement space
default_ms = np.linspace(0,10,100,False)

npbins = 25

# Default parameter space
default_ps = (
    np.linspace(0.1, 10, npbins, False), # mu
    np.linspace(0.1, 10, npbins, False), # sig
    np.linspace(10, 110, npbins, False)  # mag
)

class SGD:
    def __init__(self, learning_rate=1.0):
        self._lr = learning_rate
    def __call__(self, gradients, state):
        return -self._lr * gradients, state

class Toy:
    def __init__(self, key = random.PRNGKey(42), ms = default_ms, pses = default_ps):
        ''' Create toy calculation with random key, measurement
        linspace ms and parameter linspaces *ps '''
        self.ms = jnp.array(ms)
        self.ps = jnp.array(pses)
        self.key = key

    @cached_property
    def ps_mesh(self):
        'A meshgrid spanning parameter space'
        return jnp.meshgrid(*self.ps)

    @cached_property
    def ps_flat(self):
        'Parameter space points shaped (Npoints, Ndims)'
        return jnp.stack([one.reshape(-1) for one in self.ps_mesh]).T
    
    @property
    def qrand(self):
        'A random point in parameter space'
        q = self.ps_flat
        self.key,sub = random.split(self.key)
        ind = random.randint(sub, (), 0, q.shape[0])
        return q[ind]

    # fixme: @cache or @lru_cache?
    def predict(self, q):
        'Predict expectated central value of measurements of q' 
        mu = q[0]
        sig = q[1]
        mag = q[2]

        gnorm = mag / (jnp.sqrt(2*jnp.pi))

        d = self.ms - mu
        #print(f'predict: mu:{mu.shape}, {q.shape}->{d.shape}')

        try:
            return gnorm * jnp.exp(-0.5*((d/sig)**2))
        except FloatingPointError:
            print(f'q:{q}')
            print(f'mu:{mu}')
            print(f'ms:{ms}')
            print(f'd:{d}')
            print(f'sig:{sig}')
            print(f'gnorm:{gnorm}')
            raise

    def statv_cnp(self, N, q):
        '''
        Statistical covariance matrix following combined neyman-pearson.
        '''
        Npred = self.predict(q)
        diag = 3.0/(1.0/N + 2.0/Npred)

        diag = jnp.where(diag > 0, diag, N)
        diag = jnp.where(diag > 0, diag, 0.5*Npred)
        # see appendix a of CNP paper for the 1/2.

        c = diag * jnp.eye(diag.size)
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
        self.key, sub = random.split(self.key)
        nfluc = random.poisson(sub, npred)
        return nfluc

    def chi2(self, N, q):
        '''
        Return the chi2 value for the measurement and a prediction at q.
        '''
        print(f'chi2: q:{q.shape} N:{N.shape}')
        npred = self.predict(q)
        c = self.covariance(N,q)
        try:
            cinv = jnp.linalg.inv(c)
            dN = jnp.expand_dims(N - npred, axis=1)
            dNT = jnp.transpose(dN)
            return (dNT @ cinv @ dN).squeeze()
        except RuntimeError:
            return jnp.nan


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

        @vmap
        def Nchi2(q):
            return self.chi2(N, q)

        #Nchi2 = vmap(partial(self.chi2, N)) # over chunks of qs

        ndevs = len(jax.devices())
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

        def bydev(qschunks):
            print(f'qschunks:{qschunks.shape}')
            many = list()
            for qschunk in qschunks:
                one = Nchi2(qschunk)
                print(f'qschunk:{qschunk.shape}, one:{one.shape}')
                many.append(one)
            ret = jnp.hstack(many)
            print(f'bydev:{ret.shape}')
            return ret
        bydev = pmap(bydev)
        print('calling bydev')
        parts = bydev(qspar)
        parts = parts.reshape(-1)
        print(f'parts:{parts.shape}')
        
        leftovers = Nchi2(qs[npar:,:])
        chi2s = jnp.hstack((parts, leftovers))
        print(f'chi2s:{chi2s.shape}')
        
        nans = jnp.isnan(chi2s)
        if jnp.any(nans):
            nums = jnp.invert(nans)
            maxchi2 = jnp.max(chi2s[nums])
            print(f'NaNs in chi2, replacing with max chi2: {maxchi2}')
            bad = jnp.zeros_like(chi2s) + maxchi2
            chi2s = jnp.where(nans, bad, chi2s)
        indbest = jnp.argmin(chi2s)
        chi2best = chi2s[indbest]
        qbest = qs[indbest,:]
        print (qbest, chi2best)
        return qbest, chi2best, chi2s


    def most_likely_opt(self, N, q, opt_fn, opt_state=None, steps=100):
        '''Return parameter with the prediction that minimizes chi2 with
        the given measure N.  The parameter point q gives a starting point'''

        Nchi2 = jit(vmap(partial(self.chi2, N)))

        losses = []
        qpoints = []
        for _ in range(steps):
            loss, grads = value_and_grad(Nchi2)(q)
            updates, opt_state = opt_fn(grads, opt_state)
            q += updates
            qpoints.append(q)
            losses.append(loss)
        return jnp.stack(losses), jnp.stack(qpoints), q, opt_state
    

    def delta_chi2(self, N, p):
        '''Return the difference of two chi2 values.  First is the chi2
        value for the measurement N vs the prediction at p.  Second is the
        chi2 value for the measurement N and the prediction qbest which
        maximizes the likelihood to have produced N.
        '''
        chi2null = self.chi2(N, p)
        qbest = most_likely_gs(N) # ...fixme...
        chi2best = self.chi2(N, qbest)
        return chi2null - chi2best


###

import optax

def test_some_things(dev='cpu', chunk_size = 1000):
    # jax.config.update('jax_platform_name', 'cpu')
    # jax.config.update("jax_debug_nans", debug)
    # jax.config.update('jax_disable_jit', debug)

    t = Toy()
    q = t.qrand
    Npred = t.predict(q)
    N = t.fluctuate(q)

    c = t.covariance(N, q)
    cdiag = jnp.diag(c)
    chi2 = t.chi2(N, q)

    print('chi2(fluctuated, q) =', t.chi2(N, q))
    print('chi2(predicted, q)  =', t.chi2(Npred, q))

    # # sgd = SGD(learning_rate=1.0)
    # # losses, qbest, opt_state = most_likely(N, q, opt_fn=sgd)
    # # print ("sgd:")
    # # print (losses)
    # # print (qbest)

    # adam = optax.adam(learning_rate=1.0);
    # losses, qpionts, qbest, opt_state = most_likely(N, q, opt_fn=adam.update,
    #                                                 opt_state=adam.init(q))

    def my_func():
        qbest, chi2best, chi2s = t.most_likely_gs(N, chunk_size)
        print(f'q=\n{q}\nqbest=\n{qbest}\ndiff=\n{q-qbest}\nchi2best={chi2best}')

    import time
    start = time.perf_counter()
    my_func()
    stop = time.perf_counter()
    dt = stop-start
    print(f'elapsed={dt:.3f} s rate={len(t.ps_flat)/dt:.3f} Hz')
    return locals()

if "__main__" == __name__:
    import os
    import sys
    dev = 'gpu'
    chunk_size = 1000
    for arg in sys.argv[1:]:
        if arg.startswith("cpu"):
            dev = "cpu"
            continue
        if arg.startswith("gpu"):
            dev = 'gpu'
            continue

        try:
            chunk_size = int(arg)
        except ValueError:
            continue
            
    print(dev,chunk_size)
    print(jax.devices())
    test_some_things(dev, chunk_size)


# eg:
# python toy.py
# or:
# XLA_FLAGS='--xla_force_host_platform_device_count=10' python toy.py
