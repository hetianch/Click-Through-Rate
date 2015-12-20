'''
Python implementation of Per-Coordinate FTRL-Proximal algorithm 
@author Hetian Chen Dec 02, 2014
'''
import math
import numpy as np
from utils import sigmoid

class PCFTRL(object):
    '''
    Per-Coordinate FTRL-Proximal algorithm
    Reference: http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, lambda1, lambda2, Dim):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.Dim = Dim

        # model
        self.w = [0.] * Dim

        # helper
        self.n = [0.] * Dim
        self.z = [0.] * Dim
               

    def learn(self, x, y):
        ''' update w,z,n for one instance of x
        '''
        # parameters
        alpha = self.alpha
        beta = self.beta
        lambda1 = self.lambda1
        lambda2 = self.lambda2 

        # model
        w = self.w 

        # helper
        n = self.n
        z = self.z 
        
        # wx is the inner product of w and x
        wx = 0.
        for i in x:
            if abs(z[i]) <= lambda1:
                w[i] = 0.
            else:
            	denom = ((beta + math.sqrt(n[i])) / alpha + lambda2)
                w[i] = (np.sign(z[i]) * lambda1- z[i]) / denom

            wx += w[i]

        p = sigmoid(wx)
        g = p - y
        
        # update z and n
        for i in x:
            sigma = 1/alpha * (math.sqrt(n[i] + g ** 2) - math.sqrt(n[i])) 
            z[i] += g - sigma * w[i]
            n[i] += g ** 2

    def predict(self,x):
        wx = 0.
        w = self.w 
        for i in x:
            wx += w[i]
        p = sigmoid(wx)
        return p 


