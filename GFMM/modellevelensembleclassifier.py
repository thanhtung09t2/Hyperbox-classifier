# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:41:56 2018

@author: Thanh Tung Khuat

Model level ensemble classifiers of base GFMM-AGGLO-2

            ModelLevelEnsembleClassifier(numClassifier, numFold, gamma, teta, bthres, simil, sing, oper, isNorm, norm_range)

    INPUT
        numClassifier       The number of classifiers
        numFold             The number of folds for cross-validation
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
import sys, os
sys.path.insert(0, os.path.pardir)

import numpy as np
import time
import ast
from GFMM.basebatchlearninggfmm import BaseBatchLearningGFMM
from GFMM.accelbatchgfmm import AccelBatchGFMM
from GFMM.classification import predict
from functionhelper.prepocessinghelper import splitDatasetRndToKPart, splitDatasetRndClassBasedToKPart
from functionhelper.prepocessinghelper import loadDataset, string_to_boolean

class ModelLevelEnsembleClassifier(BaseBatchLearningGFMM):
    
    def __init__(self, numClassifier = 10, numFold = 10, gamma = 1, teta = 1, bthres = 0.5, simil = 'mid', sing = 'max', oper = 'min', isNorm = True, norm_range = [0, 1]):
        BaseBatchLearningGFMM.__init__(self, gamma, teta, False, oper, isNorm, norm_range)
        
        self.numFold = numFold
        self.numClassifier = numClassifier
        self.bthres = bthres
        self.simil = simil
        self.sing = sing
        self.numHyperboxes = 0
    
    
    def fit(self, X_l, X_u, patClassId, typeOfSplitting = 1, isRemoveContainedHyperboxes = True):
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
        
        time_start = time.clock()
        
        for i in range(self.numClassifier):
        
            if typeOfSplitting == 1:
                partitionedXtr = splitDatasetRndToKPart(X_l, X_u, patClassId, self.numFold)
            else:
                partitionedXtr = splitDatasetRndClassBasedToKPart(X_l, X_u, patClassId, self.numFold)
                
            predictor = self.training(partitionedXtr)
            
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
        
        time_end = time.clock()
        self.elapsed_training_time = time_end - time_start
        
        return self
              
    
    def training(self, partitionedXtr):
        """
        Training the base classifier using K-fold cross-validation. This method is used when the input data are preprocessed and partitioned into k parts
        
        INPUT
            partitionedXtr      An numpy array contains k sub-arrays, in which each subarray is Bunch datatype:
                                + lower:    lower bounds
                                + upper:    upper bounds
                                + label:    class labels
                                partitionedXtr should be normalized (if needed) beforehand using this function
                                
        """
        predictor = None     
        minEr = 2
        
        for k in range(self.numFold):
            predictor_tmp = AccelBatchGFMM(self.gamma, self.teta, self.bthres, self.simil, self.sing, False, self.oper, False)
            predictor_tmp.fit(partitionedXtr[k].lower, partitionedXtr[k].upper, partitionedXtr[k].label)
            
            # Create the validation set being the remaining training data
            for l in range(self.numFold):
                if l == k:
                    continue
                else:
                    if (k == 0 and l == 1) or (l == 0 and k != 0):
                        lower_valid = partitionedXtr[l].lower
                        upper_valid = partitionedXtr[l].upper
                        label_valid = partitionedXtr[l].label
                    else:
                        lower_valid = np.vstack((lower_valid, partitionedXtr[l].lower))
                        upper_valid = np.vstack((upper_valid, partitionedXtr[l].upper))
                        label_valid = np.append(label_valid, partitionedXtr[l].label)
            
            # validate the trained model
            rest = predict(predictor_tmp.V, predictor_tmp.W, predictor_tmp.classId, lower_valid, upper_valid, label_valid, self.gamma, self.oper)
            er = rest.summis / len(label_valid)
            
            if er < minEr:
                minEr = er
                predictor = predictor_tmp
            
            
        return predictor

if __name__ == '__main__':
    
    """
    INPUT parameters from command line
    arg1: + 1 - training and testing datasets are located in separated files
          + 2 - training and testing datasets are located in the same files
    arg2: path to file containing the training dataset (arg1 = 1) or both training and testing datasets (arg1 = 2)
    arg3: + path to file containing the testing dataset (arg1 = 1)
          + percentage of the training dataset in the input file
    arg4: + Number of base classifiers needs to be combined (default: 5)
    arg5: + Number of folds for cross-validation (default: 10)
    arg6: + Maximum size of hyperboxes (teta, default: 1)
    arg7: + gamma value (default: 1)
    arg8: + Similarity threshold (default: 0.5)
    arg9: + Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
    arg10: + operation used to compute membership value: 'min' or 'prod' (default: 'min')
    arg11: + do normalization of datasets or not? True: Normilize, False: No normalize (default: True)
    arg12: + range of input values after normalization (default: [0, 1])   
    arg13: + Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
    arg14: Mode to split a dataset into arg5 folds (default: 0):
            - 1: Randomly split whole dataset
            - otherwise: Randomly split following each class label
    arg16: Identify if hyperboxes contained in other hyperboxes are discarded or not? (default: True)
    """
    # Init default parameters
    if len(sys.argv) < 5:
        numBaseClassifier = 5
    else:
        numBaseClassifier = int(sys.argv[4])
    
    if len(sys.argv) < 6:
        numFold = 10    
    else:
        numFold = int(sys.argv[5])
    
    if len(sys.argv) < 7:
        teta = 1    
    else:
        teta = float(sys.argv[6])
    
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
        typeOfSplit = 0
    else:
        typeOfSplit = int(sys.argv[14])
    
    if len(sys.argv) < 16:
        isDeleteContainedHyperbox = True
    else:
        isDeleteContainedHyperbox = string_to_boolean(sys.argv[15])
        
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
   
    classifier = ModelLevelEnsembleClassifier(numClassifier = numBaseClassifier, numFold = numFold, gamma = gamma, teta = teta, bthres = bthres, simil = simil, sing = sing, oper = oper, isNorm = isNorm, norm_range = norm_range)
    print('--- Ensemble learning at model level---')
    classifier.fit(Xtr, Xtr, patClassIdTr, typeOfSplit, isDeleteContainedHyperbox)
    print('Num hyperboxes = ', classifier.numHyperboxes)
    
    # Testing
    print("-- Testing --")
    result = classifier.predict(Xtest, Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")
    
    