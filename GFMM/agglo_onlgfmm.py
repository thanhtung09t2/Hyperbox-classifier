# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:41:27 2018

@author: Thanh Tung Khuat

Another method for serial combination of online learning and agglomerative learning gfmm

    Using Agglomerative learning to train a base model, then deploy the trained model for online learning with different training data
            
            AggloOnlineGFMM(gamma, teta_onl, teta_agglo, bthres, simil, sing, isDraw, oper, isNorm, norm_range, V_pre, W_pre, classId_pre)

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
        V_pre           Hyperbox lower bounds for the model to be updated using new data
        W_pre           Hyperbox upper bounds for the model to be updated using new data
        classId_pre     Hyperbox class labels (crisp)  for the model to be updated using new data   
  
    ATTRIBUTES:
        V               Hyperbox lower bounds
        W               Hyperbox upper bounds
        classId         Hyperbox class labels (crisp)
        cardin          Hyperbox cardinalities (the number of training samples is covered by corresponding hyperboxes)
        clusters        Identifiers of input objects in each hyperbox (indexes of training samples covered by corresponding hyperboxes)

"""

import sys, os
sys.path.insert(0, os.path.pardir)

import ast
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')

from functionhelper.prepocessinghelper import loadDataset, string_to_boolean, splitDatasetRndClassBasedTo2Part, splitDatasetRndTo2Part
from basebatchlearninggfmm import BaseBatchLearningGFMM
from onlinegfmm import OnlineGFMM
from accelbatchgfmm import AccelBatchGFMM
from batchgfmm_v1 import BatchGFMMV1
from batchgfmm_v2 import BatchGFMMV2

class AggloOnlineGFMM(BaseBatchLearningGFMM):
    
    def __init__(self, gamma = 1, teta_onl = 1, teta_agglo = 1, bthres = 0.5, simil = 'mid', sing = 'max', isDraw = False, oper = 'min', isNorm = False, norm_range = [0, 1], V_pre = np.array([], dtype=np.float64), W_pre = np.array([], dtype=np.float64), classId_pre = np.array([], dtype=np.int16)):
        BaseBatchLearningGFMM.__init__(self, gamma, teta_onl, isDraw, oper, isNorm, norm_range)
        
        self.teta_onl = teta_onl
        self.teta_agglo = teta_agglo
        
        self.V = V_pre
        self.W = W_pre
        self.classId = classId_pre
        
        self.bthres = bthres
        self.simil = simil
        self.sing = sing
        
    
    def fit(self, Xl_onl, Xu_onl, patClassId_onl, Xl_off, Xu_off, patClassId_off, typeOfAgglo = 1):
        """
        The input data need to be normalized before using this function
        
        Xl_onl              Input data lower bounds (rows = objects, columns = features) for online learning
        Xu_onl              Input data upper bounds (rows = objects, columns = features) for online learning
        patClassId_onl      Input data class labels (crisp) for online learning
        
        Xl_off              Input data lower bounds (rows = objects, columns = features) for agglomerative learning
        Xu_off              Input data upper bounds (rows = objects, columns = features) for agglomerative learning
        patClassId_off      Input data class labels (crisp) for agglomerative learning
        
        typeOfAgglo         The used type of agglomerative learning algorithms
        """
        
        time_start = time.clock()
        
        # Perform agglomerative learning
        if typeOfAgglo == 1:
            aggloClassifier = AccelBatchGFMM(self.gamma, self.teta_agglo, bthres = self.bthres, simil = self.simil, sing = self.sing, isDraw = self.isDraw, oper = self.oper, isNorm = False)
        elif typeOfAgglo == 2:
            aggloClassifier = BatchGFMMV2(self.gamma, self.teta_agglo, bthres = self.bthres, simil = self.simil, sing = self.sing, isDraw = self.isDraw, oper = self.oper, isNorm = False)
        else:
            aggloClassifier = BatchGFMMV1(self.gamma, self.teta_agglo, bthres = self.bthres, simil = self.simil, sing = self.sing, isDraw = self.isDraw, oper = self.oper, isNorm = False)
            
        aggloClassifier.fit(Xl_off, Xu_off, patClassId_off)
        
        self.V = aggloClassifier.V
        self.W = aggloClassifier.W
        self.classId = aggloClassifier.classId
        
        # Perform online learning
        onlClassifier = OnlineGFMM(self.gamma, self.teta_onl, self.teta_onl, isDraw = self.isDraw, oper = self.oper, isNorm = False, norm_range = [self.loLim, self.hiLim], V = self.V, W = self.W, classId = self.classId)
        # training for online GFMM
        onlClassifier.fit(Xl_onl, Xu_onl, patClassId_onl)
        
        self.V = onlClassifier.V
        self.W = onlClassifier.W
        self.classId = onlClassifier.classId
        
        time_end = time.clock()
        self.elapsed_training_time = time_end - time_start
        
        return self
    

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
    arg14: + Type of agglomerative learning
                - 1: Accelerated agglomerative learning AGGLO-2
                - 2: Full batch learning slower version
                - 3: Full batch learning faster version
    arg15: + Percentage of online training data (default: 0.5)
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
        typeOfAgglo = 1
    else:
        typeOfAgglo = int(sys.argv[14])
        
    if len(sys.argv) < 16:
        percentOnl = 0.5
    else:
        percentOnl = float(sys.argv[15])
        
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
    
    
    classifier = AggloOnlineGFMM(gamma, teta_onl, teta_agglo, bthres, simil, sing, isDraw, oper, isNorm, norm_range)
    
    Xtr_onl, Xtr_off = splitDatasetRndTo2Part(Xtr, Xtr, patClassIdTr, percentOnl)
    
    classifier.fit(Xtr_onl.lower, Xtr_onl.upper, Xtr_onl.label, Xtr_off.lower, Xtr_off.upper, Xtr_off.label)
    
    
    # Testing
    print("-- Testing --")
    result = classifier.predict(Xtest, Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")
   
