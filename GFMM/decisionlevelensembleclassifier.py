# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 14:32:52 2018

@author: Thanh Tung Khuat

Decision level ensemble classifiers of base GFMM-AGGLO-2

            DecisionLevelEnsembleClassifier(numClassifier, gamma, teta, bthres, simil, sing, oper, isNorm, norm_range)

    INPUT
        numClassifier       The number of classifiers
        gamma               Membership function slope (default: 1)
        teta                Maximum hyperbox size (default: 1)
        bthres              Similarity threshold for hyperbox concatenetion (default: 0.5)
        simil               Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
        sing                Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
        oper                Membership calculation operation: 'min' or 'prod' (default: 'min')
        isNorm              Do normalization of input training samples or not?
        norm_range          New ranging of input data after normalization, for example: [0, 1]
        
    ATTRIBUTES
        baseClassifiers     An array of base GFMM AGLLO-2 classifiers
        numHyperboxes       The number of hyperboxes in all base classifiers
"""
import numpy as np
from basebatchlearninggfmm import BaseBatchLearningGFMM
from accelbatchgfmm import AccelBatchGFMM
from classification import predictDecisionLevelEnsemble
from prepocessinghelper import splitDatasetRndToKPart, splitDatasetRndClassBasedToKPart

class DecisionLevelEnsembleClassifier(BaseBatchLearningGFMM):
    
    def __init__(self, numClassifier = 10, gamma = 1, teta = 1, bthres = 0.5, simil = 'mid', sing = 'max', oper = 'min', isNorm = True, norm_range = [0, 1]):
        BaseBatchLearningGFMM.__init__(self, gamma, teta, False, oper, isNorm, norm_range)
        
        self.numClassifier = numClassifier
        self.bthres = bthres
        self.simil = simil
        self.sing = sing
        self.baseClassifiers = np.empty(numClassifier, dtype = BaseBatchLearningGFMM)
        self.numHyperboxes = 0
    
    
    def fit(self, X_l, X_u, patClassId, typeOfSplitting = 1):
        """
        Training the ensemble model at decision level. This method is used when the input data are not partitioned into k parts
        
        INPUT
                X_l                 Input data lower bounds (rows = objects, columns = features)
                X_u                 Input data upper bounds (rows = objects, columns = features)
                patClassId          Input data class labels (crisp)
                typeOfSplitting     The way of splitting datasets
                                        + 1: random split on whole dataset - do not care the classes
                                        + otherwise: random split according to each class label
        """
        X_l, X_u = self.dataPreprocessing(X_l, X_u)
        
        if typeOfSplitting == 1:
            partitionedXtr = splitDatasetRndToKPart(X_l, X_u, patClassId, self.numClassifier)
        else:
            partitionedXtr = splitDatasetRndClassBasedToKPart(X_l, X_u, patClassId, self.numClassifier)
            
        self.training(partitionedXtr)
            
        return self
    
    
    def training(self, partitionedXtr):
        """
        Training the ensemble model at decision level. This method is used when the input data are preprocessed and partitioned into k parts
        
        INPUT
            partitionedXtr      An numpy array contains k sub-arrays, in which each subarray is Bunch datatype:
                                + lower:    lower bounds
                                + upper:    upper bounds
                                + label:    class labels
                                partitionedXtr should be normalized (if needed) beforehand using this function
                                
        OUTPUT
            baseClassifiers     An array of classifiers needed to combine, datatype of each element in the array is BaseGFMMClassifier
            numHyperboxes       The number of hyperboxes in all base classifiers
        """
        self.numHyperboxes = 0
        
        for i in range(self.numClassifier):
            self.baseClassifiers[i] = AccelBatchGFMM(self.gamma, self.teta, self.bthres, self.simil, self.sing, False, self.oper, False)
            self.baseClassifiers[i].fit(partitionedXtr[i].lower, partitionedXtr[i].upper, partitionedXtr[i].label)
            self.numHyperboxes = self.numHyperboxes + len(self.baseClassifiers[i].classId)
        
        return (self.ensbClassifier, self.numHyperboxes)
    
    
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
                          + summis        Number of misclassified samples
                          + misclass      Binary error map for input samples
                          + out           Soft class memberships, rows are testing input patterns, columns are indices of classes
                          + classes       Store class labels corresponding column indices of out
        """
        # Normalize testing dataset if training datasets were normalized
        if len(self.mins) > 0 and self.isNorm == True:
            noSamples = Xl_Test.shape[0]
            Xl_Test = self.loLim + (self.hiLim - self.loLim) * (Xl_Test - np.ones((noSamples, 1)) * self.mins) / (np.ones((noSamples, 1)) * (self.maxs - self.mins))
            Xu_Test = self.loLim + (self.hiLim - self.loLim) * (Xu_Test - np.ones((noSamples, 1)) * self.mins) / (np.ones((noSamples, 1)) * (self.maxs - self.mins))
            
            if Xl_Test.min() < self.loLim or Xu_Test.min() < self.loLim or Xl_Test.max() > self.hiLim or Xu_Test.max() > self.hiLim:
                print('Test sample falls ousitde', self.loLim, '-', self.hiLim, 'interval')
                return
            
        # do classification
        result = predictDecisionLevelEnsemble(self.baseClassifiers, Xl_Test, Xu_Test, patClassIdTest, self.gamma, self.oper)
        
        return result
        
