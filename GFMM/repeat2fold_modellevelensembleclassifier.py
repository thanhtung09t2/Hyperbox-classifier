# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:02:04 2018

@author: Thanh Tung Khuat

    Repeated 2-fold cross-validation model level ensemble classifiers of base GFMM-AGGLO-2

            Repeat2FoldModelLevelEnsembleClassifier(numClassifier, gamma, teta, bthres, simil, sing, oper, isNorm, norm_range)

    INPUT
        numClassifier       The number of classifiers
        numFold             The number of folds for cross-validation
        gamma               Membership function slope (default: 1)
        teta                Maximum hyperbox size (default: 1)
        bthres              Initial similarity threshold for hyperbox concatenetion (default: 0.95)
        bthres_min          The minimum value of the similarity threshold (default: 0.05)
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
import ast
from basebatchlearninggfmm import BaseBatchLearningGFMM
from accelbatchgfmm import AccelBatchGFMM
from classification import predictDecisionLevelEnsemble
from functionhelper.matrixhelper import delete_const_dims
from functionhelper.prepocessinghelper import splitDatasetRndClassBasedTo2Part, splitDatasetRndTo2Part
from classification import predict
from functionhelper.prepocessinghelper import loadDataset, string_to_boolean

class Repeat2FoldModelLevelEnsembleClassifier(BaseBatchLearningGFMM):
    
    def __init__(self, numClassifier = 5, gamma = 1, teta = 1, bthres = 0.95, bthres_min = 0.05, simil = 'mid', sing = 'max', oper = 'min', isNorm = True, norm_range = [0, 1]):
        BaseBatchLearningGFMM.__init__(self, gamma, teta, False, oper, isNorm, norm_range)
        
        self.bthres_min = bthres_min
        self.numClassifier = numClassifier
        self.bthres = bthres
        self.simil = simil
        self.sing = sing
        self.baseClassifiers = np.empty(numClassifier, dtype = BaseBatchLearningGFMM)
        self.numHyperboxes = 0
    
    
    def fit(self, X_l, X_u, patClassId, typeOfSplitting = 0, training_rate = 0.5, isDeleteContainedHyperbox = True):
        """
        Training the ensemble model at decision level. This method is used when the input data are not partitioned into k parts
        
        INPUT
                X_l                         Input data lower bounds (rows = objects, columns = features)
                X_u                         Input data upper bounds (rows = objects, columns = features)
                patClassId                  Input data class labels (crisp)
                typeOfSplitting             The way of splitting datasets
                                                + 1: random split on whole dataset - do not care the classes
                                                + otherwise: random split according to each class label
                training_rate               The proportion of training data in whole dataset (the remaining data are validation data)
                isDeleteContainedHyperbox   Identify if hyperboxes contained in other hyperboxes are discarded or not?
        """
        X_l, X_u = self.dataPreprocessing(X_l, X_u)
    
        if typeOfSplitting == 1:
            (X_tr, X_val) = splitDatasetRndTo2Part(X_l, X_u, patClassId, training_rate)
        else:
            (X_tr, X_val) = splitDatasetRndClassBasedTo2Part(X_l, X_u, patClassId, training_rate)
            
        self.training(X_tr, X_val)
        
        return self
    
    
    def training(self, X_tr, X_val, isDeleteContainedHyperbox = True):
        """
        Training a base classifier using K-fold cross-validation. This method is used when the input data are preprocessed and partitioned into k parts
        
        INPUT
            X_tr       An object contains training data with the Bunch datatype, its attributes:
                        + lower:    lower bounds
                        + upper:    upper bounds
                        + label:    class labels
                        
            X_val      An object contains validation data with the Bunch datatype, its attributes:
                        + lower:    lower bounds
                        + upper:    upper bounds
                        + label:    class labels
                    X_tr, X_val should be normalized (if needed) beforehand using this function
        
            isDeleteContainedHyperbox   Identify if hyperboxes contained in other hyperboxes are discarded or not?
        """
        V_train = X_tr.lower
        W_train = X_tr.upper
        classId_train = X_tr.label
        
        V_val = X_val.lower
        W_val = X_val.upper
        classId_val = X_val.label
        
        delta_thres = (self.bthres - self.bthres_min) / self.numClassifier
        bthres = self.bthres
        self.numHyperboxes = 0
        
        N = int(self.numClassifier / 2) + 1
        
        minEr_Tr = 2
        minEr_Val = 2
        opt_Tr = None
        opt_Val = None
        
        for k in range(N):
            classifier_Tr = AccelBatchGFMM(self.gamma, self.teta, bthres, self.simil, self.sing, False, self.oper, False)
            classifier_Tr.fit(V_train, W_train, classId_train)
            
            classifier_Val = AccelBatchGFMM(self.gamma, self.teta, bthres, self.simil, self.sing, False, self.oper, False)
            classifier_Val.fit(V_val, W_val, classId_val)
            
            rest_Tr = predict(classifier_Tr.V, classifier_Tr.W, classifier_Tr.classId, V_val, W_val, classId_val, self.gamma, self.oper)
            rest_Val = predict(classifier_Val.V, classifier_Val.W, classifier_Val.classId, V_train, W_train, classId_train, self.gamma, self.oper)
            
            err_Tr = rest_Tr.summis / len(classifier_Val.classId)
            err_Val = rest_Val.summis / len(classifier_Tr.classId)            
            
            if err_Tr < minEr_Tr:
                opt_Tr = classifier_Tr
            
            if err_Val < minEr_Val:
                opt_Val = classifier_Val
              
            V_train = classifier_Tr.V
            W_train = classifier_Tr.W
            classId_train = classifier_Tr.classId
            
            V_val = classifier_Val.V
            W_val = classifier_Val.W
            classId_val = classifier_Val.classId
            
            bthres = bthres - delta_thres
        
        
        self.V = np.vstack((opt_Tr.V, opt_Val.V))
        self.W = np.vstack((opt_Tr.W, opt_Val.W))
        self.classId = np.append(opt_Tr.classId, opt_Val.classId)
        
        if isDeleteContainedHyperbox == True:
            self.removeContainedHyperboxes()
            
        self.overlapResolve()
        
        # training using AGGLO-2
        combClassifier = AccelBatchGFMM(self.gamma, self.teta, self.bthres_min, self.simil, self.sing, False, self.oper, False)
        combClassifier.fit(self.V, self.W, self.classId)
        
        self.V = combClassifier.V
        self.W = combClassifier.W
        self.classId = combClassifier.classId
        self.cardin = combClassifier.cardin
        self.clusters = combClassifier.clusters
        self.numHyperboxes = len(self.classId)
           
        return self
    
    
if __name__ == "__main__":
    
    """
    INPUT parameters from command line
    arg1: + 1 - training and testing datasets are located in separated files
          + 2 - training and testing datasets are located in the same files
    arg2: path to file containing the training dataset (arg1 = 1) or both training and testing datasets (arg1 = 2)
    arg3: + path to file containing the testing dataset (arg1 = 1)
          + percentage of the training dataset in the input file
    arg4: + Number of base classifiers needs to be combined (default: 5)
    arg5: + Maximum size of hyperboxes (teta, default: 1)
    arg6: + gamma value (default: 1)
    arg7: + Similarity threshold (default: 0.95)
    arg8: + Minimum value of Similarity threshold (default: 0.05)
    arg9: + Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
    arg10: + operation used to compute membership value: 'min' or 'prod' (default: 'min')
    arg11: + do normalization of datasets or not? True: Normilize, False: No normalize (default: True)
    arg12: + range of input values after normalization (default: [0, 1])   
    arg13: + Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
    arg14: Mode to split a dataset into arg5 folds (default: 0):
            - 1: Randomly split whole dataset
            - otherwise: Randomly split following each class label
    arg15: The proportion of training data in the input dataset
    arg16: Identify if the contained hyperboxes inside other hyperboxes are remove? (default: True)
    """
    # Init default parameters
    if len(sys.argv) < 5:
        numBaseClassifier = 5
    else:
        numBaseClassifier = int(sys.argv[4])
    
    if len(sys.argv) < 6:
        teta = 1    
    else:
        teta = float(sys.argv[5])
    
    if len(sys.argv) < 7:
        gamma = 1
    else:
        gamma = float(sys.argv[6])
    
    if len(sys.argv) < 8:
        bthres = 0.95
    else:
        bthres = float(sys.argv[7])
        
    if len(sys.argv) < 9:
        bthres_min = 0.05
    else:
        bthres_min = float(sys.argv[8])
    
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
        training_rate = 0.5
    else:
        training_rate = float(sys.argv[15])
        
    if len(sys.argv) < 16:
        isDeleteContainedHyperbox = True
    else:
        isDeleteContainedHyperbox = string_to_boolean(sys.argv[16])
        
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
    
    
    
    training_file = 'synthetic_train.dat'
    testing_file = 'synthetic_test.dat'

    # Read training file
    Xtr, X_tmp, patClassIdTr, pat_tmp = loadDataset(training_file, 1, False)
    # Read testing file
    X_tmp, Xtest, pat_tmp, patClassIdTest = loadDataset(testing_file, 0, False)
    
    classifier = Repeat2FoldModelLevelEnsembleClassifier(numClassifier = numBaseClassifier, gamma = gamma, teta = teta, bthres = bthres, bthres_min = bthres_min, simil = simil, sing = sing, oper = oper, isNorm = isNorm, norm_range = norm_range)
    print('--- Ensemble learning at model level---')
    classifier.fit(Xtr, Xtr, patClassIdTr, typeOfSplit, training_rate, isDeleteContainedHyperbox)
    print('Num hyperboxes = ', classifier.numHyperboxes)
    
    # Testing
    print("-- Testing --")
    result = classifier.predict(Xtest, Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")



