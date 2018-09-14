# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:46:02 2018

@author: Thanh Tung Khuat

Batch GFMM classifier (training core) - Slower version as mentioned in the paper

    BatchGFMMV2(gama, teta, bthres, simil, sing, isDraw, oper, isNorm, norm_range, cardin, clusters)
  
    INPUT
        gama        Membership function slope (default: 1)
        teta        Maximum hyperbox size (default: 1)
        bthres		Similarity threshold for hyperbox concatenetion (default: 0.5)
        simil       Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
        sing        Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
        oper        Membership calculation operation: 'min' or 'prod' (default: 'min')
        isDraw      Progress plot flag (default: 1)
        oper        Membership calculation operation: 'min' or 'prod' (default: 'min')
        isNorm      Do normalization of input training samples or not?
        norm_range  New ranging of input data after normalization, for example: [0, 1]
        cardin      Input hyperbox cardinalities
        clusters    Identifiers of objects in each input hyperbox 
        
    ATTRIBUTES
        V               Hyperbox lower bounds
        W               Hyperbox upper bounds
        classId         Hyperbox class labels (crisp)
        cardin          Hyperbox cardinalities (the number of training samples is covered by corresponding hyperboxes)
        clusters        Identifiers of input objects in each hyperbox (indexes of training samples covered by corresponding hyperboxes)

"""

import sys
import ast
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from basegfmmclassifier import BaseGFMMClassifier
from membershipcalc import memberG
from drawinghelper import drawbox
from hyperboxadjustment import isOverlap
from prepocessinghelper import loadDataset, string_to_boolean

class BatchGFMMV2(BaseGFMMClassifier):
    
    def __init__(self, gamma = 1, teta = 1, bthres = 0.5, simil = 'mid', sing = 'max', isDraw = False, oper = 'min', isNorm = True, norm_range = [0, 1], cardin = np.array([], dtype=np.int64), clusters = np.array([], dtype=object)):
        BaseGFMMClassifier.__init__(self, gamma, teta, isDraw, oper, isNorm, norm_range)
        
        self.bthres = bthres
        self.simil = simil
        
        if simil == 'mid':
            self.sing = sing
        else:
            self.sing = 'max'
        
        self.cardin = cardin
        self.clusters = clusters       
        
    
    def fit(self, X_l, X_u, patClassId):
        """
        X_l          Input data lower bounds (rows = objects, columns = features)
        X_u          Input data upper bounds (rows = objects, columns = features)
        patClassId  Input data class labels (crisp)
        """
        X_l, X_u = self.dataPreprocessing(X_l, X_u)
         
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
            drawing_canvas = self.initializeCanvasGraph("GFMM - AGGLO-SM-fast version", xX)
                
            # plot initial hyperbox
            Vt, Wt = self.pcatransform()
            color_ = np.empty(len(self.classId), dtype = object)
            for c in range(len(self.classId)):
                color_[c] = mark_col[self.classId[c]]
            drawbox(Vt, Wt, drawing_canvas, color_)
            self.delay()
        
        # calculate all pairwise memberships
        b = np.zeros(shape = (yX, yX))
        if self.simil == 'short':
            for j in range(yX):
                b[j, :] = memberG(self.W[j], self.V[j], self.V, self.W, self.gamma, self.oper)
        
        elif self.simil == 'long':
            for j in range(yX):
                b[j, :] = memberG(self.V[j], self.W[j], self.W, self.V, self.gamma, self.oper)
        
        else:
            for j in range(yX):
                b[j, :] = memberG(self.V[j], self.W[j], self.V, self.W, self.gamma, self.oper)
                
        maxb = self.splitSimilarityMaxtrix(b, self.sing) # get a sorted similarity (membership) list
        if len(maxb) > 0:
            maxb = maxb[maxb[:, 3] >= self.bthres, :] # scrap memberhsip values below threshold
            
        # training
        isTraining = True
        while isTraining:
            isTraining = False
            

