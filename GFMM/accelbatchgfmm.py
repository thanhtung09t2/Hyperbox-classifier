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
import sys
import ast
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from prepocessinghelper import loadDataset, string_to_boolean
from membershipcalc import asym_similarity_one_many, memberG
from drawinghelper import drawbox
from hyperboxadjustment import isOverlap
from basegfmmclassifier import BaseGFMMClassifier

class AccelBatchGFMM(BaseGFMMClassifier):
    
    def __init__(self, gamma = 1, teta = 1, bthres = 0.5, simil = 'mid', sing = 'max', isDraw = False, oper = 'min', isNorm = True, norm_range = [0, 1]):
        BaseGFMMClassifier.__init__(self, gamma, teta, isDraw, oper, isNorm, norm_range)
        
        self.bthres = bthres
        self.simil = simil
        self.sing = sing
    
    
    def fit(self, X_l, X_u, patClassId):  
        """
        Xl          Input data lower bounds (rows = objects, columns = features)
        Xu          Input data upper bounds (rows = objects, columns = features)
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
            boxes = drawbox(Vt, Wt, drawing_canvas, color_)
            plt.pause(self.delayConstant)
            hyperboxes = list(boxes)
            
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
                    b = asym_similarity_one_many(self.V[k], self.W[k], self.V, self.W, self.gamma, self.sing, self.oper)
                
                indB = np.argsort(b)[::-1]
                sortB = b[indB]
                
                maxB = sortB[sortB >= self.bthres]	# apply membership threshold
                
                if len(maxB) > 0:
                    indmaxB = indB[sortB >= self.bthres]
                    # remove self-membership
                    maxB = maxB[indmaxB != k]
                    indmaxB = indmaxB[indmaxB != k]
                    
                    # remove memberships to boxes from other classes
                    # idx_other_classes = np.where(np.logical_and(self.classId[indmaxB] != self.classId[k], self.classId[indmaxB] != 0))
                    #idx_same_classes = np.where(np.logical_or(self.classId[indmaxB] == self.classId[k], self.classId[indmaxB] == 0))
                    #idx_same_classes = np.where(self.classId[indmaxB] == self.classId[k])[0] # np.logical_or(self.classId[indmaxB] == self.classId[k], self.classId[indmaxB] == 0)
                    
                    #maxB = np.delete(maxB, idx_other_classes)
                    idx_same_classes = np.logical_or(self.classId[indmaxB] == self.classId[k], self.classId[indmaxB] == 0)
                    maxB = maxB[idx_same_classes]
                    # leaving memeberships to unlabelled boxes
                    indmaxB = indmaxB[idx_same_classes]
                    
#                    if len(maxB) > 30: # trim the set of memberships to speedup processing
#                        maxB = maxB[0:30]
#                        indmaxB = indmaxB[0:30]
                
                    pairewise_maxb = np.hstack((np.minimum(k, indmaxB)[:, np.newaxis], np.maximum(k,indmaxB)[:, np.newaxis], maxB[:, np.newaxis]))

                    for i in range(pairewise_maxb.shape[0]):
                        # calculate new coordinates of k-th hyperbox by including pairewise_maxb(i,1)-th box, scrap the latter and leave the rest intact
                        # agglomorate pairewise_maxb(i, 0) and pairewise_maxb(i, 1) by adjusting pairewise_maxb(i, 0)
                        # remove pairewise_maxb(i, 1) by getting newV from 1 -> pairewise_maxb(i, 0) - 1, new coordinates for pairewise_maxb(i, 0), from pairewise_maxb(i, 0) + 1 -> pairewise_maxb(i, 1) - 1, pairewise_maxb(i, 1) + 1 -> end
                        
                        newV = np.vstack((self.V[:int(pairewise_maxb[i, 0])], np.minimum(self.V[int(pairewise_maxb[i, 0])], self.V[int(pairewise_maxb[i, 1])]), self.V[int(pairewise_maxb[i, 0]) + 1:int(pairewise_maxb[i, 1])], self.V[int(pairewise_maxb[i, 1]) + 1:]))
                        newW = np.vstack((self.W[:int(pairewise_maxb[i, 0])], np.maximum(self.W[int(pairewise_maxb[i, 0])], self.W[int(pairewise_maxb[i, 1])]), self.W[int(pairewise_maxb[i, 0]) + 1:int(pairewise_maxb[i, 1])], self.W[int(pairewise_maxb[i, 1]) + 1:]))
                        newClassId = np.hstack((self.classId[:int(pairewise_maxb[i, 1])], self.classId[int(pairewise_maxb[i, 1]) + 1:]))
                        
                        # adjust the hyperbox if no overlap and maximum hyperbox size is not violated
                        # position of adjustment is pairewise_maxb[i, 0] in new bounds
                        if (not isOverlap(newV, newW, int(pairewise_maxb[i, 0]), newClassId)) and (((newW[int(pairewise_maxb[i, 0])] - newV[int(pairewise_maxb[i, 0])]) <= self.teta).all() == True):
                            self.V = newV
                            self.W = newW
                            self.classId = newClassId
                            
                            self.cardin[int(pairewise_maxb[i, 0])] = self.cardin[int(pairewise_maxb[i, 0])] + self.cardin[int(pairewise_maxb[i, 1])]
                            #self.cardin = np.delete(self.cardin, int(pairewise_maxb[i, 1]))
                            self.cardin = np.append(self.cardin[0:int(pairewise_maxb[i, 1])], self.cardin[int(pairewise_maxb[i, 1]) + 1:])
                            
                            self.clusters[int(pairewise_maxb[i, 0])] = np.append(self.clusters[int(pairewise_maxb[i, 0])], self.clusters[int(pairewise_maxb[i, 1])])
                            #self.clusters = np.delete(self.clusters, int(pairewise_maxb[i, 1]))
                            self.clusters = np.append(self.clusters[0:int(pairewise_maxb[i, 1])], self.clusters[int(pairewise_maxb[i, 1]) + 1:])
                            
                            isTraining = True
                            
                            if k != pairewise_maxb[i, 0]: # position pairewise_maxb[i, 1] (also k) is removed, so next step should start from pairewise_maxb[i, 1]
                                k = k - 1
                                
                            if self.isDraw:
                                try:
                                    hyperboxes[int(pairewise_maxb[i, 1])].remove()
                                    hyperboxes[int(pairewise_maxb[i, 0])].remove()
                                except:
                                    print("No remove old hyperbox")
                                
                                Vt, Wt = self.pcatransform()
                                
                                box_color = 'k'
                                if self.classId[int(pairewise_maxb[i, 0])] < len(mark_col):
                                    box_color = mark_col[self.classId[int(pairewise_maxb[i, 0])]]
                                
                                box = drawbox(np.asmatrix(Vt[int(pairewise_maxb[i, 0])]), np.asmatrix(Wt[int(pairewise_maxb[i, 0])]), drawing_canvas, box_color)
                                plt.pause(self.delayConstant)
                                hyperboxes[int(pairewise_maxb[i, 0])] = box[0]
                                hyperboxes.remove(hyperboxes[int(pairewise_maxb[i, 1])])
                                
                            break # if hyperbox adjusted there's no need to look at other hyperboxes
                            
                        
                    k = k + 1
            
        
    
if __name__ == '__main__':
    """
    INPUT parameters from command line
    arg1: + 1 - training and testing datasets are located in separated files
          + 2 - training and testing datasets are located in the same files
    arg2: path to file containing the training dataset (arg1 = 1) or both training and testing datasets (arg1 = 2)
    arg3: + path to file containing the testing dataset (arg1 = 1)
          + percentage of the training dataset in the input file
    arg4: + True: drawing hyperboxes during the training process
          + False: no drawing
    arg5: + Maximum size of hyperboxes (teta, default: 1)
    arg6: + gamma value (default: 1)
    arg7: + Similarity threshod (default: 0.5)
    arg8: + Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
    arg9: + operation used to compute membership value: 'min' or 'prod' (default: 'min')
    arg10: + do normalization of datasets or not? True: Normilize, False: No normalize (default: True)
    arg11: + range of input values after normalization (default: [0, 1])   
    arg12: + Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
    """
    # Init default parameters
    if len(sys.argv) < 5:
        isDraw = False
    else:
        isDraw = string_to_boolean(sys.argv[4])
    
    if len(sys.argv) < 6:
        teta = 1    
    else:
        teta = float(sys.argv[5])
    
    if len(sys.argv) < 7:
        gamma = 1
    else:
        gamma = float(sys.argv[6])
    
    if len(sys.argv) < 8:
        bthres = 0.5
    else:
        bthres = float(sys.argv[7])
    
    if len(sys.argv) < 9:
        simil = 'mid'
    else:
        simil = sys.argv[8]
    
    if len(sys.argv) < 10:
        oper = 'min'
    else:
        oper = sys.argv[9]
    
    if len(sys.argv) < 11:
        isNorm = True
    else:
        isNorm = string_to_boolean(sys.argv[10])
    
    if len(sys.argv) < 12:
        norm_range = [0, 1]
    else:
        norm_range = ast.literal_eval(sys.argv[11])
        
    if len(sys.argv) < 13:
        sing = 'max'
    else:
        sing = sys.argv[12]
        
    if sys.argv[1] == '1':
        training_file = sys.argv[2]
        testing_file = sys.argv[3]

        # Read training file
        Xtr, X_tmp, patClassIdTr, pat_tmp = loadDataset(training_file, 1, False)
        # Read testing file
        X_tmp, Xtest, pat_tmp, patClassIdTest = loadDataset(testing_file, 0, False)
    
    else:
        dataset_file = sys.argv[2]
        percent_Training = float(sys.argv[3])
        Xtr, Xtest, patClassIdTr, patClassIdTest = loadDataset(dataset_file, percent_Training, False)
    
    classifier = AccelBatchGFMM(gamma, teta, bthres, simil, sing, isDraw, oper, isNorm, norm_range)
    classifier.fit(Xtr, Xtr, patClassIdTr)
    
    # Testing
    print("-- Testing --")
    result = classifier.predict(Xtest, Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")