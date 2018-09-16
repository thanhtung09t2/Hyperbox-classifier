# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 10:43:55 2018

@author: Thanh Tung Khuat

Base class for batch learning GFMM
"""

import numpy as np
from basegfmmclassifier import BaseGFMMClassifier
from classification import predict

class BaseBatchLearningGFMM(BaseGFMMClassifier):
    
    def __init__(self, gamma = 1, teta = 1, isDraw = False, oper = 'min', isNorm = True, norm_range = [0, 1]):
        BaseGFMMClassifier.__init__(self, gamma, teta, isDraw, oper, isNorm, norm_range)
        
        self.V = np.array([])
        self.W = np.array([])
        self.classId = np.array([])
        self.cardin = np.array([])
        self.clusters = np.empty(None, dtype=object)


    def pruning(self, X_Val, classId_Val):
        """
        prunning routine for GFMM classifier - Hyperboxes having the number of corrected patterns lower than that of uncorrected samples are prunned
        
        INPUT
            X_Val           Validation data
            ClassId_Val     Validation data class labels (crisp)
            
        OUTPUT
            Lower and upperbounds (V and W), classId, cardin are retained
        """
        # test the model on validation data
        result = predict(self.V, self.W, self.classId, X_Val, X_Val, classId_Val, self.gamma, self.oper)
        mem = result.mem
        
        # find indexes of hyperboxes corresponding to max memberships for all validation patterns
        indmax = mem.argmax(axis = 1)
        
        numBoxes = self.V.shape[0]
        corrinc = np.zeros((numBoxes, 2))
        
        # for each hyperbox calculate the number of validation patterns classified correctly and incorrectly
        for ii in range(numBoxes):
            sampleLabelsInBox = classId_Val[indmax == ii]
            if len(sampleLabelsInBox) > 0:
                corrinc[ii, 0] = np.sum(sampleLabelsInBox == self.classId[ii])
                corrinc[ii, 1] = len(sampleLabelsInBox) - corrinc[ii, 0]
                
        # retain only the hyperboxes which classify at least the same number of patterns correctly as incorrectly
        indRetainedBoxes = np.nonzero(corrinc[:, 0] > corrinc[:, 1])[0]
        
        self.V = self.V[indRetainedBoxes, :]
        self.W = self.W[indRetainedBoxes, :]
        self.classId = self.classId[indRetainedBoxes]
        self.cardin = self.cardin[indRetainedBoxes]
        
        return self
        
        
        
        
