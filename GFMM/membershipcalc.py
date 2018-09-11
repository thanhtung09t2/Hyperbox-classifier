# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:47:45 2018

@author: Thanh Tung Khuat

Fuzzy membership calculation

"""
import numpy as np

def memberG(X_l, X_u, V, W, g, oper = 'min'):
    """
    Function for membership calculation
    
        b = memberG(X_l, X_u, V, W, g, oper)
 
   INPUT
     X_l        Input data lower bounds (a row vector with columns being features)
     X_u        Input data upper bounds (a row vector with columns being features)
     V          Hyperbox lower bounds
     W          Hyperbox upper bounds
     g          User defined sensitivity parameter 
     oper       Membership calculation operation: 'min' or 'prod' (default: 'min')
  
   OUTPUT
     b			Degrees of membership of the input pattern

   DESCRIPTION
    	Function provides the degree of membership b of an input pattern X (in form of upper bound Xu and lower bound Xl)
        in hyperboxes described by min points V and max points W. The sensitivity parameter g regulates how fast the 
        membership values decrease when an input pattern is separeted from hyperbox core.

    """
    
    yW = W.shape[0]
    
    violMax = 1 - fofmemb(np.ones((yW, 1)) * X_u - W, g)
    violMin = 1 - fofmemb(V - np.ones((yW,1)) * X_l, g)

    if oper == 'prod':
        b = np.prod(np.minimum(violMax, violMin), axis = 1)
    else:
        b = np.minimum(violMax, violMin).min(axis = 1)
    
    return b
    
    
    
def fofmemb(x, gama):

    """
    fofmemb - ramp threshold function for fuzzy membership calculation

        f = fofmemb(x,gama)
  
   INPUT
     x			Input data matrix (rows = objects, columns = attributes)
     gama		Steepness of membership function
  
   OUTPUT
     f			Fuzzy membership values

   DESCRIPTION
    	f = 1,     if x*gama > 1
    	x*gama,    if 0 =< x*gama <= 1
    	0,         if x*gama < 0
    """

    if np.size(gama) > 1: 
        p = x*(np.ones((x.shape[0], 1))*gama)
    else:
        p = x*gama;

    f = (((p >= 0) * (p <= 1)) * p + (p > 1)).astype(float);
    
    return f

def asym_similarity_one_many(Xl_k, Xu_k, V, W, g = 1, asym_oper = 'max', oper_mem = 'min'):
    """
    Calculate the asymetrical similarity value of the k-th hyperbox (lower bound - Xl_k, upper bound - Xu_k) and 
    hyperboxes having lower and upper bounds stored in two matrix V and W respectively
    
    INPUT
        Xl_k        Lower bound of the k-th hyperbox
        Xu_k        Upper bound of the k-th hyperbox
        V           Lower bounds of other hyperboxes
        W           Upper bounds of other hyperboxes
        g           User defined sensitivity parameter 
        asym_oper   Use 'min' or 'max' (default) to compute the asymetrical similarity value
        oper_mem    operator used to compute the membership value, 'min' or 'prod'
        
    OUTPUT
        b           similarity values of hyperbox k with all hyperboxes having lower and upper bounds in V and W
    
    """
    numHyperboxes = W.shape[0]
    b = np.empty(numHyperboxes, dtype = object)
    
    for k in range(numHyperboxes):
        violMax1 = 1 - fofmemb(np.ones((1, 1)) * (Xu_k - W[k]), g)
        violMin1 = 1 - fofmemb((V[k] - Xl_k) * np.ones((1,1)), g)
        
        violMax2 = 1 - fofmemb(np.ones((1, 1)) * (W[k] - Xu_k), g)
        violMin2 = 1 - fofmemb((Xl_k - V[k]) * np.ones((1,1)), g)
        
        if oper_mem == 'prod':
            b1 = np.prod(np.minimum(violMax1, violMin1), axis = 1)[0]
            b2 = np.prod(np.minimum(violMax2, violMin2), axis = 1)[0]
        else:
            b1 = np.minimum(violMax1, violMin1).min(axis = 1)[0]
            b2 = np.minimum(violMax2, violMin2).min(axis = 1)[0]
            
        if asym_oper == 'max':
            b[k] = np.maximum(b1, b2)
        else:
            b[k] = np.minimum(b1, b2)
    
    return b
