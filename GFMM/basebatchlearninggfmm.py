# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 10:43:55 2018

@author: Thanh Tung Khuat

Base class for batch learning GFMM
"""

import numpy as np
from basegfmmclassifier import BaseGFMMClassifier

class BaseBatchLearningGFMM(BaseGFMMClassifier):
    
    def __init__(self, gamma = 1, teta = 1, isDraw = False, oper = 'min', isNorm = True, norm_range = [0, 1]):
        BaseGFMMClassifier.__init__(self, gamma, teta, isDraw, oper, isNorm, norm_range)
        
        self.V = np.array([])
        self.W = np.array([])
        self.cardin = np.array([])
        self.clusters = np.empty(None, dtype=object)

