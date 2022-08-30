#!/usr/bin/env python3
'''
Define what a model must provide
'''

from abc import ABC, abstractmethod

class Base(ABC):

    @abstractmethod
    def predict(self, q, ms):
        '''Return a predicted expectation values over the measurement
        domain given point in model parameter space q.

        - q :: a point or batched points in parameter space.

        - ms :: the domain of the measurement space.

        If q is batched, so is return.

        '''
        pass

    @abstractmethod
    def covariance(self, Npred, Nmeas):
        '''Return covariance as matrix between predicted expecation
        values and measured values'''
        pass

    @abstractmethod
    def chi2(self, Npred, Nmeas, covinv):
        '''Return a chi-squared value comparing predicted expecation
        values Npred and measured values Nmeas given inverse of
        covariance matrix covinv.'''
        pass
