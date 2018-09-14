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
            maxb = maxb[maxb[:, 2] >= self.bthres, :] # scrap memberhsip values below threshold
            
        # training
        isTraining = True
        while isTraining:
            isTraining = False
            
            i = 0
            while i < maxb.shape[0]:
                # if maxb(i, 0)-th and maxb(i, 1)-th come from the same class, try to join them
                if self.classId[int(maxb[i, 0])] == self.classId[int(maxb[i, 1])]:
                    # calculate new coordinates of maxb(i,0)-th hyperbox by including maxb(i,1)-th box, scrap the latter and leave the rest intact
                    # agglomorate maxb(i, 0) and maxb(i, 1) by adjust maxb(i, 0), remove maxb(i, 1) by get newV from 1:maxb(i, 0) - 1, new coordinates for maxb(i, 0), maxb(i, 0) + 1:maxb(i, 1) - 1, maxb(i, 1) + 1:end
                    newV = np.vstack((self.V[:int(maxb[i, 0])], np.minimum(self.V[int(maxb[i, 0])], self.V[int(maxb[i, 1])]), self.V[int(maxb[i, 0]) + 1:int(maxb[i, 1])], self.V[int(maxb[i, 1]) + 1:]))
                    newW = np.vstack((self.W[:int(maxb[i, 0])], np.maximum(self.W[int(maxb[i, 0])], self.W[int(maxb[i, 1])]), self.W[int(maxb[i, 0]) + 1:int(maxb[i, 1])], self.W[int(maxb[i, 1]) + 1:]))
                    newClassId = np.hstack((self.classId[:int(maxb[i, 1])], self.classId[int(maxb[i, 1]) + 1:]))
                        
                    # adjust the hyperbox if no overlap and maximum hyperbox size is not violated
                    if (not isOverlap(newV, newW, int(maxb[i, 0]), newClassId)) and (((newW[int(maxb[i, 0])] - newV[int(maxb[i, 0])]) <= self.teta).all() == True):
                        isTraining = True
                        self.V = newV
                        self.W = newW
                        self.classId = newClassId
                        
                        self.cardin[int(maxb[i, 0])] = self.cardin[int(maxb[i, 0])] + self.cardin[int(maxb[i, 1])]
                        self.cardin = np.append(self.cardin[0:int(maxb[i, 1])], self.cardin[int(maxb[i, 1]) + 1:])
                                
                        self.clusters[int(maxb[i, 0])] = np.append(self.clusters[int(maxb[i, 0])], self.clusters[int(maxb[i, 1])])
                        self.clusters = np.append(self.clusters[0:int(maxb[i, 1])], self.clusters[int(maxb[i, 1]) + 1:])
                        
                        # recalculate all pairwise memberships
                        yX, xX = self.V.shape
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
                                
                        if self.V.shape[0] == 1:
                            maxb = np.array([])
                        else:
                            maxb = self.splitSimilarityMaxtrix(b, self.sing) # get a sorted similarity (membership) list
                            
                            if len(maxb) > 0:
                                maxb = maxb[maxb[:, 2] >= self.bthres, :]
                        if self.isDraw:
                            Vt, Wt = self.pcatransform()
                            color_ = np.empty(len(self.classId), dtype = object)
                            for c in range(len(self.classId)):
                                color_[c] = mark_col[self.classId[c]]
                            drawing_canvas.cla()
                            drawbox(Vt, Wt, drawing_canvas, color_)
                            self.delay()
                        
                        break
                        
                i = i + 1
        
        return self
    
if __name__ == '__main__':
    training_file = 'synthetic_train.dat'
    testing_file = 'synthetic_test.dat'
    
    # Read training file
    Xtr, X_tmp, patClassIdTr, pat_tmp = loadDataset(training_file, 1, False)
    # Read testing file
    X_tmp, Xtest, pat_tmp, patClassIdTest = loadDataset(testing_file, 0, False)
    
    classifier = BatchGFMMV2(1, 0.6, 0.5, 'short', 'min', True)
    classifier.fit(Xtr, Xtr, patClassIdTr)
    
    # Testing
    print("-- Testing --")
    result = classifier.predict(Xtest, Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")

