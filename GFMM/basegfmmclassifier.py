# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 10:22:19 2018

@author: Thanh Tung Khuat

Base GFMM classifier
"""
import numpy as np

from classification import predict
from matrixhelper import delete_const_dims
from prepocessinghelper import normalize

class BaseGFMMClassifier(object):
    
    def __init__(self, gamma = 1, teta = 1, isDraw = False, oper = 'min', isNorm = True, norm_range = [0, 1]):
        self.gamma = gamma
        self.teta = teta
        self.isDraw = isDraw
        self.oper = oper
        self.isNorm = isNorm
      
        # parameters for data normalization
        self.loLim = norm_range[0]
        self.hiLim = norm_range[1]
        self.mins = []
        self.maxs = []
        self.delayConstant = 0.001 # delay time period to display hyperboxes on the canvas
    
    def dataPreprocessing(self, X_l, X_u):
        """
        Preprocess data: delete constant dimensions, Normalize input samples if needed
        
        INPUT:
            X_l          Input data lower bounds (rows = objects, columns = features)
            X_u          Input data upper bounds (rows = objects, columns = features)
        
        OUTPUT
            X_l, X_u were preprocessed
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
            
        return (X_l, X_u)
    
    
    def predict(self, Xl_Test, Xu_Test, patClassIdTest):
        """
        Perform classification
        
            result = predict(Xl_Test, Xu_Test, patClassIdTest)
        
        INPUT:
            Xl_Test             Test data lower bounds (rows = objects, columns = features)
            Xu_Test             Test data upper bounds (rows = objects, columns = features)
            patClassIdTest	     Test data class labels (crisp)
            
        OUTPUT:
            result        A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + sumamb           Number of objects with maximum membership in more than one class
                          + out              Soft class memberships
                          + mem              Hyperbox memberships
        """
        # Normalize testing dataset if training datasets were normalized
        if len(self.mins) > 0:
            noSamples = Xl_Test.shape[0]
            Xl_Test = self.loLim + (self.hiLim - self.loLim) * (Xl_Test - np.ones((noSamples, 1)) * self.mins) / (np.ones((noSamples, 1)) * (self.maxs - self.mins))
            Xu_Test = self.loLim + (self.hiLim - self.loLim) * (Xu_Test - np.ones((noSamples, 1)) * self.mins) / (np.ones((noSamples, 1)) * (self.maxs - self.mins))
            
            if Xl_Test.min() < self.loLim or Xu_Test.min() < self.loLim or Xl_Test.max() > self.hiLim or Xu_Test.max() > self.hiLim:
                print('Test sample falls ousitde', self.loLim, '-', self.hiLim, 'interval')
                return
            
        # do classification
        result = predict(self.V, self.W, self.classId, Xl_Test, Xu_Test, patClassIdTest, self.gamma, self.oper)
        
        return result
