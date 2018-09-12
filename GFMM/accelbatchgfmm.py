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
#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matrixhelper import delete_const_dims, pca_transform
from prepocessinghelper import normalize, loadDataset
from membershipcalc import asym_similarity_one_many, memberG
from drawinghelper import drawbox
from hyperboxadjustment import isOverlap
from classification import predict

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
        self.delayConstant = 0.001 # delay time period to display hyperboxes on the canvas
    
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
            fig = plt.figure("GFMM - AGGLO-2")
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
            boxes = drawbox(Vt, Wt, plt, color_)
            plt.pause(self.delayConstant)
            hyperboxes = list(boxes)
            
        # training
        isTraining = True
        while isTraining:
            isTraining = False
            
            k = 0 # input pattern index
            while k < len(self.classId):
                if self.simil == 'short':
                    b = memberG(np.maximum(self.W[k], self.V[k]), np.minimum(self.V[k], self.W[k]), np.minimum(self.V, self.W), np.maximum(self.W, self.V), self.gamma, self.oper)
                elif self.simil == 'long':
                    b = memberG(self.V[k], self.W[k], self.W, self.V, self.gamma, self.oper)
                else:
                    b = asym_similarity_one_many(self.V[k], self.W[k], self.V, self.W, self.gamma, self.sing, self.oper)
                
                indB = np.argsort(b)[::-1]
                sortB = b[indB]
                maxB = sortB[sortB >= self.bthres]	# apply membership threshold
                
                if len(maxB) > 0:
                    indmaxB = indB[sortB >= self.bthres]
                    # remove self-membership
                    idx_k = np.where(indmaxB == k)[0]
                    maxB = np.delete(maxB, idx_k)
                    indmaxB = np.delete(indmaxB, idx_k)
                    
                    # remove memberships to boxes from other classes
                    idx_other_classes = np.where(np.logical_and(self.classId[indmaxB] != self.classId[k], self.classId[indmaxB] != 0))
                    maxB = np.delete(maxB, idx_other_classes)
                    # leaving memeberships to unlabelled boxes
                    indmaxB = np.delete(indmaxB, idx_other_classes)
                
                    pairewise_maxb = np.hstack((np.minimum(k, indmaxB).reshape(-1, 1), np.maximum(k,indmaxB).reshape(-1, 1), maxB.reshape(-1, 1)))

                    for i in range(pairewise_maxb.shape[0]):
                        # calculate new coordinates of k-th hyperbox by including pairewise_maxb(i,1)-th box, scrap the latter and leave the rest intact
                        # agglomorate pairewise_maxb(i, 0) and pairewise_maxb(i, 1) by adjusting pairewise_maxb(i, 0)
                        # remove pairewise_maxb(i, 1) by getting newV from 1 -> pairewise_maxb(i, 0) - 1, new coordinates for pairewise_maxb(i, 0), from pairewise_maxb(i, 0) + 1 -> pairewise_maxb(i, 1) - 1, pairewise_maxb(i, 1) + 1 -> end
                        
                        newV = np.vstack((self.V[0:pairewise_maxb[i, 0].astype(np.int64)], np.minimum(self.V[pairewise_maxb[i, 0].astype(np.int64)], self.V[pairewise_maxb[i, 1].astype(np.int64)]), self.V[pairewise_maxb[i, 0].astype(np.int64) + 1:pairewise_maxb[i, 1].astype(np.int64)], self.V[pairewise_maxb[i, 1].astype(np.int64) + 1:]))
                        newW = np.vstack((self.W[0:pairewise_maxb[i, 0].astype(np.int64)], np.maximum(self.W[pairewise_maxb[i, 0].astype(np.int64)], self.W[pairewise_maxb[i, 1].astype(np.int64)]), self.W[pairewise_maxb[i, 0].astype(np.int64) + 1:pairewise_maxb[i, 1].astype(np.int64)], self.W[pairewise_maxb[i, 1].astype(np.int64) + 1:]))
                        newClassId = np.hstack((self.classId[0:pairewise_maxb[i, 1].astype(np.int64)], self.classId[pairewise_maxb[i, 1].astype(np.int64) + 1:]))
                        
                        # adjust the hyperbox if no overlap and maximum hyperbox size is not violated
                        # position of adjustment is pairewise_maxb[i, 0] in new bounds
                        if not isOverlap(newV, newW, pairewise_maxb[i, 0].astype(np.int64), newClassId) and ((newW[pairewise_maxb[i, 0].astype(np.int64), :] - newV[pairewise_maxb[i, 0].astype(np.int64),:]) <= self.teta).all() == True:
                            self.V = newV
                            self.W = newW
                            self.classId = newClassId
                            
                            self.cardin[pairewise_maxb[i, 0].astype(np.int64)] = self.cardin[pairewise_maxb[i, 0].astype(np.int64)] + self.cardin[pairewise_maxb[i, 1].astype(np.int64)]
                            self.cardin = np.delete(self.cardin, pairewise_maxb[i, 1].astype(np.int64))
                            
                            self.clusters[pairewise_maxb[i, 0].astype(np.int64)] = np.append(self.clusters[pairewise_maxb[i, 0].astype(np.int64)], self.clusters[pairewise_maxb[i, 1].astype(np.int64)])
                            self.clusters = np.delete(self.clusters, pairewise_maxb[i, 1].astype(np.int64))
                            
                            isTraining = True
                            
                            if k != pairewise_maxb[i, 0]:
                                k = k - 1
                                
                            if self.isDraw:
                                try:
                                    hyperboxes[pairewise_maxb[i, 1].astype(np.int64)].remove()
                                    hyperboxes[pairewise_maxb[i, 0].astype(np.int64)].remove()
                                except:
                                    pass
                                
                                Vt, Wt = self.pcatransform()
                                
                                box_color = 'k'
                                if self.classId[pairewise_maxb[i, 0].astype(np.int64)] < len(mark_col):
                                    box_color = mark_col[self.classId[pairewise_maxb[i, 0].astype(np.int64)]]
                                
                                box = drawbox(np.asmatrix(Vt[pairewise_maxb[i, 0].astype(np.int64)]), np.asmatrix(Wt[pairewise_maxb[i, 0].astype(np.int64)]), drawing_canvas, box_color)
                                plt.pause(self.delayConstant)
                                hyperboxes[pairewise_maxb[i, 0].astype(np.int64)] = box[0]
                                hyperboxes.remove(hyperboxes[pairewise_maxb[i, 1].astype(np.int64)])
                                
                            break # if hyperbox adjusted there's no need to look at other hyperboxes
                            
                        
                k = k + 1
            
        
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
    
if __name__ == '__main__':
    training_file = 'synthetic_train.dat'
    testing_file = 'synthetic_test.dat'

    # Read training file
    Xtr, X_tmp, patClassIdTr, pat_tmp = loadDataset(training_file, 1, False)
    # Read testing file
    X_tmp, Xtest, pat_tmp, patClassIdTest = loadDataset(testing_file, 0, False)
    
    classifier = AccelBatchGFMM(1, 0.6, 0.5, 'short', 'min', True, 'min', False, [0, 1])
    classifier.fit(Xtr, Xtr, patClassIdTr)
    
    # Testing
    print("-- Testing --")
    result = classifier.predict(Xtest, Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")