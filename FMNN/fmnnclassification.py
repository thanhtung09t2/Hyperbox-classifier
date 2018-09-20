# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 14:00:25 2018

@author: Thanh Tung Khuat

Implementation of the original fuzzy min-max neural network


"""
import numpy as np

from basefmnnclassifier import BaseFMNNClassifier

class FMNNClassification(BaseFMNNClassifier):
    
    def __init__(self, gamma = 1, teta = 1, isDraw = False, isNorm = False, norm_range = [0, 1], V = np.array([], dtype=np.float64), W = np.array([], dtype=np.float64), classId = np.array([], dtype=np.int16)):
        BaseFMNNClassifier.__init__(self, gamma, teta, isDraw, isNorm, norm_range)
        
        self.V = V
        self.W = W
        self.classId = classId