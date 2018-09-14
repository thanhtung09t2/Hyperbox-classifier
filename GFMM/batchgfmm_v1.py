# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 14:57:55 2018

@author: Khuat Thanh Tung

Batch GFMM classifier (training core) - Faster version by only computing similarity among hyperboxes with the same label

    BatchGFMM(gama, teta, bthres, simil, sing, isDraw, oper, isNorm, norm_range, cardin, clusters)
  
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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from basegfmmclassifier import BaseGFMMClassifier
from membershipcalc import memberG
from drawinghelper import drawbox
from hyperboxadjustment import isOverlap
from prepocessinghelper import loadDataset, string_to_boolean

class BatchGFMMV1(BaseGFMMClassifier):
    
    def __init__(self, gamma = 1, teta = 1, bthres = 0.5, simil = 'mid', sing = 'max', isDraw = False, oper = 'min', isNorm = True, norm_range = [0, 1], cardin = np.array([], dtype=np.int64), clusters = np.array([], dtype=object)):
        BaseGFMMClassifier.__init__(self, gamma, teta, isDraw, oper, isNorm, norm_range)
        
        self.bthres = bthres
        self.simil = simil
        self.sing = sing   
        
        self.cardin = cardin
        self.clusters = clusters
        
    
    def splitSimilarityMaxtrix(self, A, asimil_type = 'max', isSort = True):
        """
        Split the similarity matrix A into the maxtrix with three columns:
            + First column is row indices of A
            + Second column is column indices of A
            + Third column is the values corresponding to the row and column
        
        if isSort = True, the third column is sorted in the descending order 
        
            INPUT
                A               Degrees of membership of input patterns (each row is the output from memberG function)
                asimil_type     Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
                isSort          Sorting flag
                
            OUTPUT
                The output as mentioned above
        """
        # get min/max memberships from triu and tril of memberhsip matrix which might not be symmetric (simil=='mid')
        if asimil_type == 'min':
            transformedA = np.minimum(np.flipud(np.rot90(np.tril(A, -1))), np.triu(A, 1))  # rotate tril to align it with triu for min (max) operation
        else:
            transformedA = np.maximum(np.flipud(np.rot90(np.tril(A, -1))), np.triu(A, 1))
        
        ind_rows, ind_columns = np.nonzero(transformedA)
        values = A[ind_rows, ind_columns]
        
        if isSort == True:
            ind_SortedTransformedA = np.argsort(values)[::-1]
            sortedTransformedA = values[ind_SortedTransformedA]
            result = np.hstack((ind_rows[ind_SortedTransformedA][:, np.newaxis], ind_columns[ind_SortedTransformedA][:, np.newaxis], sortedTransformedA[:, np.newaxis]))
        else:
            result = np.hstack((ind_rows[:, np.newaxis], ind_columns[:, np.newaxis], values[:, np.newaxis]))
            
        return result
        
    
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
            fig = plt.figure("GFMM - AGGLO-SM-fast version")
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
            Vt, Wt = self.pcatransform()
            color_ = np.empty(len(self.classId), dtype = object)
            for c in range(len(self.classId)):
                color_[c] = mark_col[self.classId[c]]
            drawbox(Vt, Wt, drawing_canvas, color_)
            plt.pause(self.delayConstant)
        
        # training
        isTraining = True
        while isTraining:
            isTraining = False
            
            # calculate class masks
            yX, xX = self.V.shape
            labList = np.unique(self.classId)
            clMask = np.zeros(shape = (yX, len(labList)), dtype = np.bool)
            for i in range(len(labList)):
                clMask[:, i] = self.classId == labList[i]
        
        	# calculate pairwise memberships *ONLY* within each class (faster!)
            b = np.zeros(shape = (yX, yX), dtype = np.bool)
            
            for i in range(len(labList)):
                Vi = self.V[clMask[:, i]] # get bounds of patterns with class label i
                Wi = self.W[clMask[:, i]]
                clSize = np.sum(clMask[:, i]) # get number of patterns of class i
                clIdxs = np.nonzero(clMask[:, i])[0] # get position of patterns with class label i in the training set
                
                if self.simil == 'short':
                    for j in range(clSize):
                        b[clIdxs[j], clIdxs] = memberG(Wi[j], Vi[j], Vi, Wi, self.gamma, self.oper)
                elif self.simil == 'long':
                    for j in range(clSize):
                        b[clIdxs[j], clIdxs] = memberG(Vi[j], Wi[j], Wi, Vi, self.gamma, self.oper)
                else:
                    for j in range(clSize):
                        b[clIdxs[j], clIdxs] = memberG(Vi[j], Wi[j], Vi, Wi, self.gamma, self.oper)
                
            if yX == 1:
                maxb = np.array([])
            else:
                maxb = self.splitSimilarityMaxtrix(b, self.sing, True)
                if len(maxb) > 0:
                    maxb = maxb[(maxb[:, 2] >= self.bthres), :]
                    
                    if len(maxb) > 0:
                        # sort maxb in the decending order following the last column
                        idx_smaxb = np.argsort(maxb[:, 2])[::-1]
                        maxb = np.hstack((maxb[idx_smaxb, 0].reshape(-1, 1), maxb[idx_smaxb, 1].reshape(-1, 1), maxb[idx_smaxb, 2].reshape(-1, 1)))
                        #maxb = maxb[idx_smaxb]
            
            while len(maxb) > 0:
                curmaxb = maxb[0, :] # current position handling
                
                # calculate new coordinates of curmaxb(0)-th hyperbox by including curmaxb(1)-th box, scrap the latter and leave the rest intact
                newV = np.vstack((self.V[0:int(curmaxb[0]), :], np.minimum(self.V[int(curmaxb[0]), :], self.V[int(curmaxb[1]), :]), self.V[int(curmaxb[0]) + 1:int(curmaxb[1]), :], self.V[int(curmaxb[1]) + 1:, :]))
                newW = np.vstack((self.W[0:int(curmaxb[0]), :], np.maximum(self.W[int(curmaxb[0]), :], self.W[int(curmaxb[1]), :]), self.W[int(curmaxb[0]) + 1:int(curmaxb[1]), :], self.W[int(curmaxb[1]) + 1:, :]))
                newClassId = np.hstack((self.classId[0:int(curmaxb[1])], self.classId[int(curmaxb[1]) + 1:]))
                
                # adjust the hyperbox if no overlap and maximum hyperbox size is not violated
                if (not isOverlap(newV, newW, int(curmaxb[0]), newClassId)) and (((newW[int(curmaxb[0])] - newV[int(curmaxb[0])]) <= self.teta).all() == True):
                    isTraining = True
                    self.V = newV
                    self.W = newW
                    self.classId = newClassId
                    
                    self.cardin[int(curmaxb[0])] = self.cardin[int(curmaxb[0])] + self.cardin[int(curmaxb[1])]
                    self.cardin = np.append(self.cardin[0:int(curmaxb[1])], self.cardin[int(curmaxb[1]) + 1:])
                            
                    self.clusters[int(curmaxb[0])] = np.append(self.clusters[int(curmaxb[0])], self.clusters[int(curmaxb[1])])
                    self.clusters = np.append(self.clusters[0:int(curmaxb[1])], self.clusters[int(curmaxb[1]) + 1:])
                    
                    # remove joined pair from the list as well as any pair with lower membership and consisting of any of joined boxes
                    mask = (maxb[:, 0] != int(curmaxb[0])) & (maxb[:, 1] != int(curmaxb[0])) & (maxb[:, 0] != int(curmaxb[1])) & (maxb[:, 1] != int(curmaxb[1])) & (maxb[:, 2] >= curmaxb[2])
                    maxb = maxb[mask, :]
                    
                    # update indexes to accomodate removed hyperbox
                    # indices of V and W larger than curmaxb(1,2) are decreased 1 by the element whithin the location curmaxb(1,2) was removed 
                    #if len(maxb) > 0:
                    maxb[maxb[:, 0] > int(curmaxb[1]), 0] = maxb[maxb[:, 0] > int(curmaxb[1]), 0] - 1
                    maxb[maxb[:, 1] > int(curmaxb[1]), 1] = maxb[maxb[:, 1] > int(curmaxb[1]), 1] - 1
                            
                    if self.isDraw:
                        Vt, Wt = self.pcatransform()
                        color_ = np.empty(len(self.classId), dtype = object)
                        for c in range(len(self.classId)):
                            color_[c] = mark_col[self.classId[c]]
                        drawing_canvas.cla()
                        drawbox(Vt, Wt, drawing_canvas, color_)
                        plt.pause(self.delayConstant)
                else:
                    maxb = maxb[1:, :]  # scrap examined pair from the list

                  
if __name__ == '__main__':
    training_file = 'synthetic_train.dat'
    testing_file = 'synthetic_test.dat'
    
    # Read training file
    Xtr, X_tmp, patClassIdTr, pat_tmp = loadDataset(training_file, 1, False)
    # Read testing file
    X_tmp, Xtest, pat_tmp, patClassIdTest = loadDataset(testing_file, 0, False)
    
    classifier = BatchGFMMV1(1, 0.6, 0.5, 'short', 'min', True)
    classifier.fit(Xtr, Xtr, patClassIdTr)
    
    # Testing
    print("-- Testing --")
    result = classifier.predict(Xtest, Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")                