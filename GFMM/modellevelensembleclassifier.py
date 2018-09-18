# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:41:56 2018

@author: Thanh Tung Khuat

Model level ensemble classifiers of base GFMM-AGGLO-2

            ModelLevelEnsembleClassifier(numClassifier, gamma, teta, bthres, simil, sing, oper, isNorm, norm_range)

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
        V               Hyperbox lower bounds
        W               Hyperbox upper bounds
        classId         Hyperbox class labels (crisp)
        cardin          Hyperbox cardinalities (the number of training samples is covered by corresponding hyperboxes)
        clusters        Identifiers of input objects in each hyperbox (indexes of training samples covered by corresponding hyperboxes)

"""

import numpy as np
from basebatchlearninggfmm import BaseBatchLearningGFMM
from accelbatchgfmm import AccelBatchGFMM
from prepocessinghelper import splitDatasetRndToKPart, splitDatasetRndClassBasedToKPart
from hyperboxadjustment import hyperboxOverlapTest, hyperboxContraction
from membershipcalc import memberG

class ModelLevelEnsembleClassifier(BaseBatchLearningGFMM):
    
    def __init__(self, numClassifier = 10, gamma = 1, teta = 1, bthres = 0.5, simil = 'mid', sing = 'max', oper = 'min', isNorm = True, norm_range = [0, 1]):
        BaseBatchLearningGFMM.__init__(self, gamma, teta, False, oper, isNorm, norm_range)
        
        self.numClassifier = numClassifier
        self.bthres = bthres
        self.simil = simil
        self.sing = sing
        self.numHyperboxes = 0
    
    
    def fit(self, X_l, X_u, patClassId, typeOfSplitting = 1, isRemoveContainedHyperboxes = False):
        """
        Training the ensemble model at decision level. This method is used when the input data are not partitioned into k parts
        
        INPUT
                X_l                 Input data lower bounds (rows = objects, columns = features)
                X_u                 Input data upper bounds (rows = objects, columns = features)
                patClassId          Input data class labels (crisp)
                typeOfSplitting     The way of splitting datasets
                                        + 1: random split on whole dataset - do not care the classes
                                        + otherwise: random split according to each class label
                isRemoveContainedHyperboxes:  Identify if hyperboxes contained in other hyperboxes are discarded or not?
        """
        X_l, X_u = self.dataPreprocessing(X_l, X_u)
        
        if typeOfSplitting == 1:
            partitionedXtr = splitDatasetRndToKPart(X_l, X_u, patClassId, self.numClassifier)
        else:
            partitionedXtr = splitDatasetRndClassBasedToKPart(X_l, X_u, patClassId, self.numClassifier)
            
        self.training(partitionedXtr, isRemoveContainedHyperboxes)
            
        return self
    
    
    def overlapResolve(self):
        """
        Resolve overlapping hyperboxes with bounders contained in self.V and self.W
        """
        yX = self.V.shape[0]
        # Contraction process does not cause overlappling regions => No need to check from the first hyperbox for each hyperbox
        for i in np.arange(yX - 1):
            j = i + 1
            while j < yX:
                caseDim = hyperboxOverlapTest(self.V, self.W, i, j)
                if len(caseDim) > 0 and self.classId[i] != self.classId[j]:
                    self.V, self.W = hyperboxContraction(self.V, self.W, caseDim, j, i)
                
                j = j + 1
                
        return (self.V, self.W)
                
      
    def removeContainedHyperboxes(self):
        """
        Remove all hyperboxes contained in other hyperboxes
        """
        numBoxes = len(self.classId)
        indtokeep = np.ones(numBoxes, dtype=np.bool)
        
        for i in range(numBoxes):
            memValue = memberG(self.V[i], self.W[i], self.V, self.W, self.gamma, self.oper)
            isInclude = (self.classId[memValue == 1] == self.classId[i]).all()
            
            # memValue always has one value being 1 because of self-containing
            if np.sum(memValue == 1) > 1 and isInclude == True:
                indtokeep[i] = False
                
        self.V = self.V[indtokeep, :]
        self.W = self.W[indtokeep, :]
        self.classId = self.classId[indtokeep]
        self.clusters = self.clusters[indtokeep]
        self.cardin = self.cardin[indtokeep]
        
        
    
    def training(self, partitionedXtr, isRemoveContainedHyperboxes = False):
        """
        Training the ensemble model at decision level. This method is used when the input data are preprocessed and partitioned into k parts
        
        INPUT
            partitionedXtr      An numpy array contains k sub-arrays, in which each subarray is Bunch datatype:
                                + lower:    lower bounds
                                + upper:    upper bounds
                                + label:    class labels
                                partitionedXtr should be normalized (if needed) beforehand using this function
                                
        """
        self.numHyperboxes = 0
        
        for i in range(self.numClassifier):
            predictor = AccelBatchGFMM(self.gamma, self.teta, self.bthres, self.simil, self.sing, False, self.oper, False)
            predictor.fit(partitionedXtr[i].lower, partitionedXtr[i].upper, partitionedXtr[i].label)
            
            if i == 0:
                self.V = predictor.V
                self.W = predictor.W
                self.classId = predictor.classId
                self.cardin = predictor.cardin
                self.clusters = predictor.clusters
            else:
                self.V = np.vstack((self.V, predictor.V))
                self.W = np.vstack((self.W, predictor.W))
                self.classId = np.append(self.classId, predictor.classId)
                self.cardin = np.append(self.cardin, predictor.cardin)
                self.clusters = np.append(self.clusters, predictor.clusters)
                
            
        if isRemoveContainedHyperboxes == True:
            self.removeContainedHyperboxes()
            
        self.overlapResolve()
        
        # training using AGGLO-2
        combClassifier = AccelBatchGFMM(self.gamma, self.teta, self.bthres, self.simil, self.sing, False, self.oper, False)
        combClassifier.fit(self.V, self.W, self.classId)
        
        self.V = combClassifier.V
        self.W = combClassifier.W
        self.classId = combClassifier.classId
        self.cardin = combClassifier.cardin
        self.clusters = combClassifier.clusters
        self.numHyperboxes = len(self.classId)

        return self

