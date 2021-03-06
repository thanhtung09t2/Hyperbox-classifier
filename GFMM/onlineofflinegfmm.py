# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 09:41:27 2018

@author: Thanh Tung Khuat

Simple combination of online learning and agglomerative learning gfmm
            
            OnlineAggloGFMM(gamma, teta_onl, teta_agglo, bthres, simil, sing, isDraw, oper, isNorm, norm_range, V_pre, W_pre, classId_pre)

    INPUT
        gamma           Membership function slope (default: 1)
        teta_onl        Maximum hyperbox size (default: 1) for online learning
        teta_agglo      Maximum hyperbox size (default: 1) for agglomerative v2 learning
        bthres          Similarity threshold for hyperbox concatenation (default: 0.5)
        simil           Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
        sing            Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
        isDraw          Progress plot flag (default: False)
        oper            Membership calculation operation: 'min' or 'prod' (default: 'min')
        isNorm          Do normalization of input training samples or not?
        norm_range      New ranging of input data after normalization, for example: [0, 1]
        
    ATTRIBUTES:
        onlClassifier   online classifier with the following attributes:
                            + V: hyperbox lower bounds
                            + W: hyperbox upper bounds
                            + classId: hyperbox class labels (crisp)
                            
        offClassifier   offline classifier with the following attributes:
                            + V: hyperbox lower bounds
                            + W: hyperbox upper bounds
                            + classId: hyperbox class labels (crisp)
"""

import sys, os
sys.path.insert(0, os.path.pardir)

import ast
import time
import numpy as np
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

from functionhelper.preprocessinghelper import loadDataset, string_to_boolean, splitDatasetRndClassBasedTo2Part, splitDatasetRndTo2Part
from functionhelper.matrixhelper import delete_const_dims
from functionhelper.bunchdatatype import Bunch
from GFMM.basebatchlearninggfmm import BaseBatchLearningGFMM
from GFMM.onlinegfmm import OnlineGFMM
from GFMM.accelbatchgfmm import AccelBatchGFMM
from GFMM.classification import predictOnlineOfflineCombination

class OnlineOfflineGFMM(BaseBatchLearningGFMM):
    
    def __init__(self, gamma = 1, teta_onl = 1, teta_agglo = 1, bthres = 0.5, simil = 'mid', sing = 'max', isDraw = False, oper = 'min', isNorm = True, norm_range = [0, 1]):
        BaseBatchLearningGFMM.__init__(self, gamma, teta_onl, isDraw, oper, isNorm, norm_range)
        
        self.teta_onl = teta_onl
        self.teta_agglo = teta_agglo
        
        self.bthres = bthres
        self.simil = simil
        self.sing = sing
        
    
    def fit(self, Xl_onl, Xu_onl, patClassId_onl, Xl_off, Xu_off, patClassId_off):
        """
        Input data need to be normalized before using this function
        
        Xl_onl              Input data lower bounds (rows = objects, columns = features) for online learning
        Xu_onl              Input data upper bounds (rows = objects, columns = features) for online learning
        patClassId_onl      Input data class labels (crisp) for online learning
        
        Xl_off              Input data lower bounds (rows = objects, columns = features) for agglomerative learning
        Xu_off              Input data upper bounds (rows = objects, columns = features) for agglomerative learning
        patClassId_off      Input data class labels (crisp) for agglomerative learning
        """
        
        time_start = time.clock()
        # Perform agglomerative learning
        aggloClassifier = AccelBatchGFMM(self.gamma, self.teta_agglo, bthres = self.bthres, simil = self.simil, sing = self.sing, isDraw = self.isDraw, oper = self.oper, isNorm = False)
        aggloClassifier.fit(Xl_off, Xu_off, patClassId_off)
        self.offClassifier = Bunch(V = aggloClassifier.V, W = aggloClassifier.W, classId = aggloClassifier.classId)
        
        # Perform online learning
        onlClassifier = OnlineGFMM(self.gamma, self.teta_onl, self.teta_onl, isDraw = self.isDraw, oper = self.oper, isNorm = False, norm_range = [self.loLim, self.hiLim])
        onlClassifier.fit(Xl_onl, Xu_onl, patClassId_onl)   
        self.onlClassifier = Bunch(V = onlClassifier.V, W = onlClassifier.W, classId = onlClassifier.classId)
        
        time_end = time.clock()
        self.elapsed_training_time = time_end - time_start
        
        return self

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
        #Xl_Test, Xu_Test = delete_const_dims(Xl_Test, Xu_Test)
        # Normalize testing dataset if training datasets were normalized
        if len(self.mins) > 0:
            noSamples = Xl_Test.shape[0]
            Xl_Test = self.loLim + (self.hiLim - self.loLim) * (Xl_Test - np.ones((noSamples, 1)) * self.mins) / (np.ones((noSamples, 1)) * (self.maxs - self.mins))
            Xu_Test = self.loLim + (self.hiLim - self.loLim) * (Xu_Test - np.ones((noSamples, 1)) * self.mins) / (np.ones((noSamples, 1)) * (self.maxs - self.mins))
            
            if Xl_Test.min() < self.loLim or Xu_Test.min() < self.loLim or Xl_Test.max() > self.hiLim or Xu_Test.max() > self.hiLim:
                print('Test sample falls outside', self.loLim, '-', self.hiLim, 'interval')
                print('Number of original samples = ', noSamples)
                
                # only keep samples within the interval loLim-hiLim
                indXl_good = np.where((Xl_Test >= self.loLim).all(axis = 1) & (Xl_Test <= self.hiLim).all(axis = 1))[0]
                indXu_good = np.where((Xu_Test >= self.loLim).all(axis = 1) & (Xu_Test <= self.hiLim).all(axis = 1))[0]
                indKeep = np.intersect1d(indXl_good, indXu_good)
                
                Xl_Test = Xl_Test[indKeep, :]
                Xu_Test = Xu_Test[indKeep, :]
                
                print('Number of kept samples =', Xl_Test.shape[0])
                #return
            
        # do classification
        result = None
        
        if Xl_Test.shape[0] > 0:
            result = predictOnlineOfflineCombination(self.onlClassifier, self.offClassifier, Xl_Test, Xu_Test, patClassIdTest, self.gamma, self.oper)
        
        return result  

if __name__ == '__main__':
    """
    INPUT parameters from command line
    
    arg1:  + 1 - training and testing datasets are located in separated files
           + 2 - training and testing datasets are located in the same files
    arg2:  path to file containing the training dataset (arg1 = 1) or both training and testing datasets (arg1 = 2)
    arg3:  + path to file containing the testing dataset (arg1 = 1)
           + percentage of the training dataset in the input file
    arg4:  + True: drawing hyperboxes during the training process
           + False: no drawing
    arg5:  + Maximum size of hyperboxes of online learning algorithm (teta_onl, default: 1)
    arg6:  + Maximum size of hyperboxes of agglomerative learning algorithm (teta_agglo, default: 1)
    arg7:  + gamma value (default: 1)
    arg8:  + Similarity threshod (default: 0.5)
    arg9:  + Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
    arg10: + operation used to compute membership value: 'min' or 'prod' (default: 'min')
    arg11: + do normalization of datasets or not? True: Normilize, False: No normalize (default: True)
    arg12: + range of input values after normalization (default: [0, 1])   
    arg13: + Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
    arg14: + Percentage of online training data (default: 0.5)

    """
    
    # Init default parameters
    if len(sys.argv) < 5:
        isDraw = False
    else:
        isDraw = string_to_boolean(sys.argv[4])
    
    if len(sys.argv) < 6:
        teta_onl = 1    
    else:
        teta_onl = float(sys.argv[5])
    
    if len(sys.argv) < 7:
        teta_agglo = 1
    else:
        teta_agglo = float(sys.argv[6])
    
    if len(sys.argv) < 8:
        gamma = 1
    else:
        gamma = float(sys.argv[7])
    
    if len(sys.argv) < 9:
        bthres = 0.5
    else:
        bthres = float(sys.argv[8])
    
    if len(sys.argv) < 10:
        simil = 'mid'
    else:
        simil = sys.argv[9]
    
    if len(sys.argv) < 11:
        oper = 'min'
    else:
        oper = sys.argv[10]
    
    if len(sys.argv) < 12:
        isNorm = True
    else:
        isNorm = string_to_boolean(sys.argv[11])
    
    if len(sys.argv) < 13:
        norm_range = [0, 1]
    else:
        norm_range = ast.literal_eval(sys.argv[12])
        
    if len(sys.argv) < 14:
        sing = 'max'
    else:
        sing = sys.argv[13]
        
    if len(sys.argv) < 15:
        percentOnl = 0.5
    else:
        percentOnl = float(sys.argv[14])
        
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
    
    
    classifier = OnlineOfflineGFMM(gamma, teta_onl, teta_agglo, bthres, simil, sing, isDraw, oper, isNorm, norm_range)
    
    Xtr_onl, Xtr_off = splitDatasetRndTo2Part(Xtr, Xtr, patClassIdTr, percentOnl)
    
    classifier.fit(Xtr_onl.lower, Xtr_onl.upper, Xtr_onl.label, Xtr_off.lower, Xtr_off.upper, Xtr_off.label)
    
    # Testing
    print("-- Testing --")
    result = classifier.predict(Xtest, Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")
   
