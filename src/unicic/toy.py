#!/usr/bin/env python3
'''
A toy model for testing unicic
'''

from unicic import model

class Toy(model.Base):

    def __init__(self, low):
        '''
        Create a toy model.

        - low :: one of modules in unicic.low.*

        '''
        self.low = low
        self.xp = low.xp


    def predict(self, q, ms):
        ''' Predict expected number of events to be measured in bins
        over the measurement space given Gaussian parameters q.'''
        q = self.xp.array(q)
        ms = self.xp.array(ms)

        mu,sig,mag = q.T            # same for scalar or batched

        gnorm = mag / (self.xp.sqrt(2*self.xp.pi))

        d = ms.reshape(1,-1) - mu.reshape(-1,1) # (nbatch,nbins)
        pred = gnorm.reshape(-1, 1) * self.xp.exp(-0.5*((d/sig.reshape(-1,1))**2))
        qdims = len(q.shape)
        if qdims == 1:
            pred = pred.reshape((ms.size,))
        return pred


    def statvar_cnp_diag(self, Npred, Nmeas):
        '''Return the diagonal of the statistical covariance matrix
        following "combined Nyeman-Pearson" construction with additional
        protection for zeros and infinites.

        Npred and Nmeas may be any mix of scalar shape (nbins,) or batched
        (nbatch, nbins).  If either are batched, the return is batched.
        If both are batched, they must be batched the same size and a
        batch-to-batch comparison is make.

        Npred must be nonzero everywhere.
        '''

        num = 3.0 * Nmeas * Npred
        den = 2.0 * Nmeas + Npred

        good_meas = Nmeas > 0
        num = self.xp.where(good_meas, num, 0.5*Npred)
        den = self.xp.where(good_meas, den, 1.0)

        # good_both = self.xp.logical_and(Npred > 0, good_meas)
        # num = self.xp.where(good_both, num, Nmeas)
        # den = self.xp.where(good_both, den, 1)

        if not self.xp.all(num > 0):
            print (f'\nNpred:\n{Npred}')
            print (f'\nNmeas:\n{Nmeas}')
            print (f'num:\n{num}')
            print (f'den:\n{den}')
            raise ValueError("zeros found on statistical covariance diagonal")

        diag = num/den
        return diag

    def statvar_cnp(self, Npred, Nmeas):
        '''Statistical part of the covariance.'''
        diag = self.statvar_cnp_diag(Npred, Nmeas)
        diag = self.xp.expand_dims(diag, axis=1)
        I = self.xp.eye(Nmeas.shape[-1])
        return diag * I

    def covariance(self, Npred, Nmeas):
        return self.statvar_cnp(Npred, Nmeas)


    def chi2(self, Npred, Nmeas, covinv):
        '''Return the chi2 value between the expectation value of
        predicted measure Npred and an actual measure Nmeas and
        INVERSE of a covariance matrix invcov.

        Npred and Nmeas may be any mix of scalar shape (nbins,) or
        batched (nbatch, nbins), etc for convinv (nbins,nbins) or
        (nbatch,nbins,nbins).  If any are batched then so is the
        return.  Any batching must be matched.  If the N's are not
        batched but the covinv is, the calculation will work, however,
        this may not be a meaninful thing to do.

        '''
        # see test_tmm.test_ein for sussing this out
    
        # get batched dimensions
        pdims = len(Npred.shape)-1
        cdims = len(covinv.shape)-2
        mdims = len(Nmeas.shape)-1
        odims = max((pdims,cdims,mdims))

        einp = 'B'*pdims + 'i'
        einc = 'B'*cdims + 'ij'
        einm = 'B'*mdims + 'j'
        eino = 'B'*odims

        ein = f'{einp},{einc},{einm} -> {eino}'
        ret = self.xp.einsum(ein, Npred, covinv, Nmeas)
        return ret

    
    
