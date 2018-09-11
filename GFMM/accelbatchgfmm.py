# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 09:42:14 2018

@author: Thanh Tung Khuat

    Accelerated Batch GFMM classifier (training core)
        
        AccelBatchGFMM(gama, teta, bthres, simil, sing, isDraw, oper, isNorm, norm_range)
  
    INPUT:
        gama            Membership function slope (default: 1)
        teta            Maximum hyperbox size (default: 1)
        bthres          Similarity threshold for hyperbox concatenetion (default: 0.5)
        simil           Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
        sing            Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
        isDraw          Progress plot flag (default: False)
        oper            Membership calculation operation: 'min' or 'prod' (default: 'min')
        isNorm          Do normalization of input training samples or not?
        norm_range      New ranging of input data after normalization, for example: [0, 1]
  
    ATTRIBUTES:
        V               Hyperbox lower bounds
        W               Hyperbox upper bounds
        classId         Hyperbox class labels (crisp)
        cardin          Hyperbox cardinalities (the number of training samples is covered by corresponding hyperboxes)
        clusters        Identifiers of input objects in each hyperbox (indexes of training samples covered by corresponding hyperboxes)

"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matrixhelper import delete_const_dims, pca_transform
from prepocessinghelper import normalize
from membershipcalc import asym_similarity_one_many, memberG
from drawinghelper import drawbox

class AccelBatchGFMM(object):
    
    def __init__(self, gamma = 1, teta = 1, bthres = 0.5, simil = 'mid', sing = 'max', isDraw = False, oper = 'min', isNorm = True, norm_range = [0, 1]):
        self.gamma = gamma
        self.teta = teta
        self.bthres = bthres
        self.simil = simil
        self.sing = sing
        self.isDraw = isDraw
        self.oper = oper
        self.isNorm = isNorm
        self.loLim = norm_range[0]
        self.hiLim = norm_range[1]
        self.mins = []
        self.maxs = []
    
    def pcatransform(self):
        """
        Perform PCA transform of V and W if the dimensions are larger than 3
        
        OUTPUT:
            V and W in the new space
        """
        yX, xX = self.V.shape
                
        if (xX > 3):
            Vt = pca_transform(self.V, 3)
            Wt = pca_transform(self.W, 3)
            mins = Vt.min(axis = 0)
            maxs = Wt.max(axis = 0)
            Vt = self.loLim + (self.hiLim - self.loLim) * (Vt - np.ones((yX, 1)) * mins) / (np.ones((yX, 1)) * (maxs - mins))
            Wt = self.loLim + (self.hiLim - self.loLim) * (Wt - np.ones((yX, 1)) * mins) / (np.ones((yX, 1)) * (maxs - mins))
        else:
            Vt = self.V
            Wt = self.W
            
        return (Vt, Wt)
    
    
    def fit(self, X_l, X_u, patClassId):  
        """
        Xl          Input data lower bounds (rows = objects, columns = features)
        Xu          Input data upper bounds (rows = objects, columns = features)
        patClassId  Input data class labels (crisp)
        """
        # delete constant dimensions
        X_l, X_u = delete_const_dims(X_l, X_u)
        
        # Normalize input samples if needed
        if X_l.min() < self.loLim or X_u.min() < self.loLim or X_u.max() > self.hiLim or X_l.max() > self.hiLim:
            self.mins = X_l.min(axis = 0) # get min value of each feature
            self.maxs = X_u.max(axis = 0) # get max value of each feature
            X_l = normalize(X_l, [self.loLim, self.hiLim])
            X_u = normalize(X_u, [self.loLim, self.hiLim])
        else:
            self.isNorm = False
            self.mins = []
            self.maxs = []
         
        self.V = X_l
        self.W = X_u
        self.classId = patClassId
        
        yX, xX = X_l.shape
        self.cardin = np.ones(yX)
        self.clusters = np.empty(yX, dtype=object)
        for i in range(yX):
            self.clusters[i] = np.array([i], dtype = np.int32)
        
        if self.isDraw:
            mark_col = np.array(['r', 'g', 'b', 'y', 'c', 'm', 'k'])
            fig = plt.figure(0)
            plt.ion()
            if xX == 2:
                drawing_canvas = fig.add_subplot(1, 1, 1)
                drawing_canvas.axis([0, 1, 0, 1])
            else:
                drawing_canvas = Axes3D(fig)
                drawing_canvas.set_xlim3d(0, 1)
                drawing_canvas.set_ylim3d(0, 1)
                drawing_canvas.set_zlim3d(0, 1)
                
            # plot initial hyperbox
            Vt, Wt = pcatransform()
            color_ = np.empty(len(self.classId), dtype = object)
            for c in range(len(self.classId)):
                color_[c] = mark_col[self.classId[c]]
            hyperboxes = drawbox(Vt, Wt, plt, color_)
            
        # training
        isTraining = True
        while isTraining:
            isTraining = False
            
            k = 0 # input pattern index
            while k < len(self.classId):
                if self.simil == 'short':
                    b = memberG(self.W[k], self.V[k], self.V, self.W, self.gamma, self.oper)
                elif self.simil == 'long':
                    b = memberG(self.V[k], self.W[k], self.W, self.V, self.gamma, self.oper)
                else:
                    b = asym_similarity_one_many(self.V[k], self.W[k], self.V, self.W, self.g, self.sing, self.oper)
                
            
        # remove self-membership
        # idx_k = np.where(indmaxB == k)[0]
        #maxB = np.delete(maxB, idx_k)
        #indmaxB = np.delete(indmaxb, idx_k)
        #maxb = np.hstack((np.minimum(k, indmaxB).reshape(-1, 1), np.maximum(k,indmaxB).reshape(-1, 1), maxB.reshape(-1, 1)))
            
      
                
            
        # remove self-membership
        # idx_k = np.where(indmaxB == k)[0]
        #maxB = np.delete(maxB, idx_k)
        #indmaxB = np.delete(indmaxb, idx_k)
        #maxb = np.hstack((np.minimum(k, indmaxB).reshape(-1, 1), np.maximum(k,indmaxB).reshape(-1, 1), maxB.reshape(-1, 1)))
            
      
    
    