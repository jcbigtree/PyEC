"""Benchmark functions. 

Benchmark functions for single or multi-objective real-parameter. optimization
"""

import numpy as np


class SCHFunction(object):
    """
    Attributes:
    -----------
    dim_ : 1
    upper_bound_ : np.array(1000)
    lower_bound_ : np.array(-1000)
    
    Objectives:
    -----------
    f1 = x^2
    f2 = (x-2)^2
    """
    def __init__(self):
        self.dim_ = 1 
        self.upper_bound_ = np.array([1000])
        self.lower_bound_ = np.array([-1000])
    
    
    def __call__(self, x):
        f1 = np.power(x,2)
        f2 = np.power(x-2,2)
        return f1, f2 
        
###################################################################################################    
   
class ZDT2Function(object):
    """
    Attributes:
    -----------
    dim_ : 30
    upper_bound_ : np.array([1.0]*30)
    lower_bound_ : np.array([-1.0]*30)
    
    Objectives:
    -----------
    f1 = x1
    f2 = (x-2)^2
    """
    def __init__(self):
        self.dim_ = 5
        self.upper_bound_ = np.array([1]*self.dim_)
        self.lower_bound_ = np.array([0]*self.dim_)
    
    
    def __call__(self, x):
        return self.ZDT2(x) 
    
    
    def ZDT2(self, x):        
        g = x[1:].sum()/(self.dim_-1.0)*9.0+1.0
        f1 = x[0]
        f2 = g*(1.0 - (x[0]/g)*(x[0]/g))
        return f1, f2
        
    
###################################################################################################
class SphereFunction(object):
    """Sphere function. sum of the square of each dimension
    
    Parameters:
    -----------
    x : array-like. -5.12 < x(i) < 5.12
    
    Attributes:
    -----------
    dim_ : 10 
     
    upper_bound_ : np.array([5.12]*10)    
    
    lower_bound_ : np.array([-5.12]*10)
    """
    def __init__(self):
        self.dim_ = 10 
        self.upper_bound_ = np.array([5.12]*10)
        self.lower_bound_ = np.array([-5.12]*10)
    
    
    def __call__(self, x):
        return - self._sphere_function(x)
        
    
    def _sphere_function(self,x):
        x = np.asmatrix(x)
        x = x.reshape((-1,1))
        return (x.transpose() * x)[0,0]
    

###################################################################################################
class EasomFunction(object):
    """Easom function. Minimum : f(pi, pi) = 1
    
    Parameters:
    -----------
    x. array-like [x,y]. 0 < x(i) < 100. 

    Attributes:
    -----------
    dim_ : 2
    
    upper_bound_ : [100, 100]
    
    lower_bound_ : [-100, -100]
    """ 
    def __init__(self):
        self.dim_ = 2
        self.upper_bound_ = np.array([100,100])
        self.lower_bound_ = np.array([-100,-100])
       
       
    def __call__(self,x):
        return -self._easom_function(x)
    
    
    def _easom_function(self,x):
        y = - np.cos(x[0])*np.cos(x[1])*np.exp( \
                    -(x[0]-np.pi)*(x[0]-np.pi)-(x[1]-np.pi)*(x[1]-np.pi) \
                    )
        return y
    
    
    

    
