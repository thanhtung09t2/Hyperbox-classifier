# -*- coding: utf-8 -*-
"""
Created on November 06

@author: Thanh Tung Khuat

    Accelerated Batch GFMM classifier (training core)
    
    Implemented by Pytorch
        
        AccelBatchGFMM(gamma, teta, bthres, simil, sing, isDraw, oper, isNorm, norm_range)
  
    INPUT:
        gamma           Membership function slope (default: 1)
        teta            Maximum hyperbox size (default: 1)
        bthres          Similarity threshold for hyperbox concatenation (default: 0.5)
        simil           Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
        sing            Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
        isDraw          Progress plot flag (default: False)
        oper            Membership calculation operation: 'min' or 'prod' (default: 'min')
        isNorm          Do normalization of input training samples or not?
        norm_range      New ranging of input data after normalization, for example: [0, 1]
        cardin      Input hyperbox cardinalities
        clusters    Identifiers of objects in each input hyperbox 
        
    ATTRIBUTES:
        V               Hyperbox lower bounds
        W               Hyperbox upper bounds
        classId         Hyperbox class labels (crisp)
        # Comment 2 attributes to accelerate the code because of non-using now
        cardin          Hyperbox cardinalities (the number of training samples is covered by corresponding hyperboxes)
        clusters        Identifiers of input objects in each hyperbox (indexes of training samples covered by corresponding hyperboxes)

"""

import sys, os
sys.path.insert(0, os.path.pardir)

import ast
import numpy as np
import torch
import time
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

from functionhelper.preprocessinghelper import loadDataset, string_to_boolean
from functionhelper.drawinghelper import drawbox
from functionhelper.torch_hyperboxadjustment import torch_isOverlap
from GFMM.torch_basegfmmclassifier import Torch_BaseGFMMClassifier
from functionhelper.torch_membership_calc import torch_asym_similarity_one_many, torch_memberG, gpu_memberG
from functionhelper import is_Have_GPU, GPU_Computing_Threshold

class Torch_AccelBatchGFMM(Torch_BaseGFMMClassifier):
    
    def __init__(self, gamma = 1, teta = 1, bthres = 0.5, simil = 'mid', sing = 'max', isDraw = False, oper = 'min', isNorm = True, norm_range = [0, 1]):
        Torch_BaseGFMMClassifier.__init__(self, gamma, teta, isDraw, oper, isNorm, norm_range)
        
        self.bthres = bthres
        self.simil = simil
        self.sing = sing
        
        # Currently, we do not yet use cardin and clusters
#        self.cardin = cardin
#        self.clusters = clusters
    
    
    def fit(self, X_l, X_u, patClassId):  
        """
        Xl          Input data lower bounds (rows = objects, columns = features)
        Xu          Input data upper bounds (rows = objects, columns = features)
        patClassId  Input data class labels (crisp)
        """
        
        if self.isNorm == True:
            X_l, X_u = self.dataPreprocessing(X_l, X_u)
        
        if isinstance(X_l, torch.Tensor) == False:
            X_l = torch.from_numpy(X_l).float()
            X_u = torch.from_numpy(X_u).float()
            patClassId = torch.from_numpy(patClassId).long()
            
        time_start = time.perf_counter()
        
        isUsingGPU = False
        if is_Have_GPU and X_l.size(0) * X_l.size(1) >= GPU_Computing_Threshold:
            self.V = X_l.cuda()
            self.W = X_u.cuda()
            self.classId = patClassId.cuda()
            isUsingGPU = True
        else:
            self.V = X_l
            self.W = X_u
            self.classId = patClassId
         
        # yX, xX = X_l.shape
        
#        if len(self.cardin) == 0 or len(self.clusters) == 0:
#            self.cardin = np.ones(yX)
#            self.clusters = np.empty(yX, dtype=object)
#            for i in range(yX):
#                self.clusters[i] = np.array([i], dtype = np.int64)
        
        if self.isDraw:
            mark_col = np.array(['r', 'g', 'b', 'y', 'c', 'm', 'k'])
            drawing_canvas = self.initializeCanvasGraph("GFMM - AGGLO-2")
                
            # plot initial hyperbox
            Vt, Wt = self.pcatransform()
            color_ = np.empty(len(self.classId), dtype = object)
            for c in range(len(self.classId)):
                color_[c] = mark_col[self.classId[c]]
            boxes = drawbox(Vt, Wt, drawing_canvas, color_)
            self.delay()
            hyperboxes = list(boxes)
            
        # training
        isTraining = True
        while isTraining:
            isTraining = False
            
            k = 0 # input pattern index
            while k < len(self.classId):
                if self.simil == 'short':
                    if isUsingGPU == False:
                        b = torch_memberG(self.W[k], self.V[k], self.V, self.W, self.gamma, self.oper, isUsingGPU)
                    else:
                        b = gpu_memberG(self.W[k], self.V[k], self.V, self.W, self.gamma, self.oper)
                        
                elif self.simil == 'long':
                    if isUsingGPU == False:
                        b = torch_memberG(self.V[k], self.W[k], self.W, self.V, self.gamma, self.oper, isUsingGPU)
                    else:
                        b = gpu_memberG(self.V[k], self.W[k], self.W, self.V, self.gamma, self.oper)
                        
                else:
                    b = torch_asym_similarity_one_many(self.V[k], self.W[k], self.V, self.W, self.gamma, self.sing, self.oper, isUsingGPU)
                
                sortB, indB = torch.sort(b, descending=True)
                
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
                    idx_same_classes = (self.classId[indmaxB] == self.classId[k]) | (self.classId[indmaxB] == 0)
                    maxB = maxB[idx_same_classes]
                    # leaving memeberships to unlabelled boxes
                    indmaxB = indmaxB[idx_same_classes]
                    
#                    if len(maxB) > 30: # trim the set of memberships to speedup processing
#                        maxB = maxB[0:30]
#                        indmaxB = indmaxB[0:30]
                    if isUsingGPU == True:
                        kMat = torch.cuda.LongTensor([k]).expand(indmaxB.size(0))
                    else:
                        kMat = torch.LongTensor([k]).expand(indmaxB.size(0))
                    
                    pairewise_maxb = torch.cat((torch.min(kMat, indmaxB).reshape(-1, 1).float(), torch.max(kMat,indmaxB).reshape(-1, 1).float(), maxB.reshape(-1, 1)), dim=1)

                    if isUsingGPU:
                        els = torch.arange(pairewise_maxb.size(0)).cuda()
                    else:
                        els = torch.arange(pairewise_maxb.size(0))
                        
                    for i in els:
                        # calculate new coordinates of k-th hyperbox by including pairewise_maxb(i,1)-th box, scrap the latter and leave the rest intact
                        # agglomorate pairewise_maxb(i, 0) and pairewise_maxb(i, 1) by adjusting pairewise_maxb(i, 0)
                        # remove pairewise_maxb(i, 1) by getting newV from 1 -> pairewise_maxb(i, 0) - 1, new coordinates for pairewise_maxb(i, 0), from pairewise_maxb(i, 0) + 1 -> pairewise_maxb(i, 1) - 1, pairewise_maxb(i, 1) + 1 -> end
                        
                        newV = torch.cat((self.V[:pairewise_maxb[i, 0].long()], torch.min(self.V[pairewise_maxb[i, 0].long()], self.V[pairewise_maxb[i, 1].long()]).reshape(1, -1), self.V[pairewise_maxb[i, 0].long() + 1:pairewise_maxb[i, 1].long()], self.V[pairewise_maxb[i, 1].long() + 1:]), dim=0)
                        newW = torch.cat((self.W[:pairewise_maxb[i, 0].long()], torch.max(self.W[pairewise_maxb[i, 0].long()], self.W[pairewise_maxb[i, 1].long()]).reshape(1, -1), self.W[pairewise_maxb[i, 0].long() + 1:pairewise_maxb[i, 1].long()], self.W[pairewise_maxb[i, 1].long() + 1:]), dim=0)
                        newClassId = torch.cat((self.classId[:pairewise_maxb[i, 1].long()], self.classId[pairewise_maxb[i, 1].long() + 1:]))
                        
                        # adjust the hyperbox if no overlap and maximum hyperbox size is not violated
                        # position of adjustment is pairewise_maxb[i, 0] in new bounds
                        if (not torch_isOverlap(newV, newW, pairewise_maxb[i, 0].long(), newClassId)) and (((newW[pairewise_maxb[i, 0].long()] - newV[pairewise_maxb[i, 0].long()]) <= self.teta).all() == True):
                            self.V = newV
                            self.W = newW
                            self.classId = newClassId
                            
#                            self.cardin[int(pairewise_maxb[i, 0])] = self.cardin[int(pairewise_maxb[i, 0])] + self.cardin[int(pairewise_maxb[i, 1])]
#                            #self.cardin = np.delete(self.cardin, int(pairewise_maxb[i, 1]))
#                            self.cardin = np.append(self.cardin[0:int(pairewise_maxb[i, 1])], self.cardin[int(pairewise_maxb[i, 1]) + 1:])
#                            
#                            self.clusters[int(pairewise_maxb[i, 0])] = np.append(self.clusters[int(pairewise_maxb[i, 0])], self.clusters[int(pairewise_maxb[i, 1])])
#                            #self.clusters = np.delete(self.clusters, int(pairewise_maxb[i, 1]))
#                            self.clusters = np.append(self.clusters[0:int(pairewise_maxb[i, 1])], self.clusters[int(pairewise_maxb[i, 1]) + 1:])
#                            
                            isTraining = True
                            
                            if k != pairewise_maxb[i, 0]: # position pairewise_maxb[i, 1] (also k) is removed, so next step should start from pairewise_maxb[i, 1]
                                k = k - 1
                                
                            if self.isDraw:
                                try:
                                    hyperboxes[pairewise_maxb[i, 1].long()].remove()
                                    hyperboxes[pairewise_maxb[i, 0].long()].remove()
                                except:
                                    print("No remove old hyperbox")
                                
                                Vt, Wt = self.pcatransform()
                                
                                box_color = 'k'
                                if self.classId[pairewise_maxb[i, 0].long()] < len(mark_col):
                                    box_color = mark_col[self.classId[pairewise_maxb[i, 0].long()]]
                                
                                box = drawbox(np.asmatrix(Vt[pairewise_maxb[i, 0].long()]), np.asmatrix(Wt[pairewise_maxb[i, 0].long()]), drawing_canvas, box_color)
                                self.delay()
                                hyperboxes[pairewise_maxb[i, 0].long()] = box[0]
                                hyperboxes.remove(hyperboxes[pairewise_maxb[i, 1].long()])
                                
                            break # if hyperbox adjusted there's no need to look at other hyperboxes
                            
                        
                k = k + 1
                    
            if isTraining == True and isUsingGPU == True and self.V.size(0) * self.V.size(1) < GPU_Computing_Threshold:
                isUsingGPU = False
                self.V = self.V.cpu()
                self.W = self.W.cpu()
                self.classId = self.classId.cpu()
        
        time_end = time.perf_counter()
        self.elapsed_training_time = time_end - time_start
         
        return self
            
        
    
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
    arg7: + Similarity threshold (default: 0.5)
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
    
    classifier = Torch_AccelBatchGFMM(gamma, teta, bthres, simil, sing, isDraw, oper, isNorm, norm_range)
    classifier.fit(Xtr, Xtr, patClassIdTr)
    
    # Testing
    print("-- Testing --")
    result = classifier.predict_torch(Xtest, Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")
        print("Training time = ", classifier.elapsed_training_time)