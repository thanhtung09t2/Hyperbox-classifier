# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:18:22 2018

@author: Thanh Tung Khuat

Preprocessing functions helper

"""

import numpy as np
from functionhelper.bunchdatatype import Bunch

def normalize(A, new_range):
    """
    Normalize the input dataset
    
    INPUT
        A           Original dataset (numpy array) [rows are samples, cols are features]
        new_range   The range of data after normalizing
   
    OUTPUT
        Normalized dataset
    """
    D = A.copy()
    n, m = D.shape
    
    for i in range(m):
        v = D[:, i]
        minv = v.min()
        maxv = v.max()
        
        if minv == maxv:
            v = np.ones(n) * 0.5;
        else:
            v = new_range[0] + (new_range[1] - new_range[0]) * (v - minv) / (maxv - minv)
        
        D[:, i] = v;
    
    return D


def loadDataset(path, percentTr, isNorm = False, new_range = [0, 1]):
    """
    Load file containing dataset and convert data in the file to training and testing datasets. Class labels are located in the last column in the file
    
        Xtr, Xtest, patClassIdTr, patClassIdTest = loadDataset(path, percentTr, True, [0, 1])
    
    INPUT
       path             the path to the data file (including file name)
       percentTr        the percentage of data used for training (0 <= percentTr <= 1)
       isNorm           identify whether normalizing datasets or not, True => Normalized
       new_range        new range of datasets after normalization

    OUTPUT
       Xtr              Training dataset
       Xtest            Testing dataset
       patClassIdTr     Training class labels
       patClassIdTest   Testing class labels
       
    """
    
    lstData = []
    with open(path) as f:
        for line in f:
            nums = np.fromstring(line.rstrip('\n').replace(',', ' '), dtype=np.float32, sep=' ').tolist()
            lstData.append(nums)
#            if (a.size == 0):
#                a = nums.reshape(1, -1)
#            else:
#                a = np.concatenate((a, nums.reshape(1, -1)), axis=0)
    A = np.array(lstData, dtype=np.float32)
    YA, XA = A.shape
   
    X_data = A[:, 0:XA-1]
    classId_dat = A[:, -1];
    classLabels = np.unique(classId_dat)
    
    # class labels must start from 1, class label = 0 means no label
    if classLabels.size > 1 and np.size(np.nonzero(classId_dat == 0)) > 0:
        classId_dat = classId_dat + 1
        classLabels = classLabels + 1

    if isNorm:
        X_data = normalize(X_data, new_range)
    
    if percentTr != 1 and percentTr != 0:
        noClasses = classLabels.size
        
        Xtr = np.empty((0, XA - 1), dtype=np.float32)
        Xtest = np.empty((0, XA - 1), dtype=np.float32)

        patClassIdTr = np.array([], dtype=np.int16)
        patClassIdTest = np.array([], dtype=np.int16)
    
        for k in range(noClasses):
            idx = np.nonzero(classId_dat == classLabels[k])[0]
            # randomly shuffle indices of elements belonging to class classLabels[k]
            if percentTr != 1 and percentTr != 0:
                idx = idx[np.random.permutation(len(idx))] 
    
            noTrain = int(len(idx) * percentTr + 0.5)
    
            # Attach data of class k to corresponding datasets
            Xtr_tmp = X_data[idx[0:noTrain], :]
            Xtr = np.concatenate((Xtr, Xtr_tmp), axis=0)
            patClassId_tmp = np.full(noTrain, classLabels[k], dtype=np.int16)
            patClassIdTr = np.append(patClassIdTr, patClassId_tmp)
            
            patClassId_tmp = np.full(len(idx) - noTrain, classLabels[k], dtype=np.int16)
            Xtest = np.concatenate((Xtest, X_data[idx[noTrain:len(idx)], :]), axis=0)
            patClassIdTest = np.concatenate((patClassIdTest, patClassId_tmp))
        
    else:
        if percentTr == 1:
            Xtr = X_data
            patClassIdTr = np.array(classId_dat, dtype=np.int64)
            Xtest = np.array([])
            patClassIdTest = np.array([])
        else:
            Xtr = np.array([])
            patClassIdTr = np.array([])
            Xtest = X_data
            patClassIdTest = np.array(classId_dat, dtype=np.int64)
        
    return (Xtr, Xtest, patClassIdTr, patClassIdTest)

def loadDatasetWithoutClassLabel(path, percentTr, isNorm = False, new_range = [0, 1]):
    """
    Load file containing dataset without class label and convert data in the file to training and testing datasets.
    
        Xtr, Xtest = loadDatasetWithoutClassLabel(path, percentTr, True, [0, 1])
    
    INPUT
       path             the path to the data file (including file name)
       percentTr        the percentage of data used for training (0 <= percentTr <= 1)
       isNorm           identify whether normalizing datasets or not, True => Normalized
       new_range        new range of datasets after normalization

    OUTPUT
       Xtr              Training dataset
       Xtest            Testing dataset
       
    """
    lstData = []
    with open(path) as f:
        for line in f:
            nums = np.fromstring(line.rstrip('\n').replace(',', ' '), dtype=np.float32, sep=' ').tolist()
            lstData.append(nums)
#            if (X_data.size == 0):
#                X_data = nums.reshape(1, -1)
#            else:
#                X_data = np.concatenate((X_data, nums.reshape(1, -1)), axis = 0)
    X_data = np.array(lstData, dtype=np.float32)
    if isNorm:
        X_data = normalize(X_data, new_range)
        
    # randomly shuffle indices of elements in the dataset
    numSamples = X_data.shape[0]
    newInds = np.random.permutation(numSamples)
    
    if percentTr != 1 and percentTr != 0:
        noTrain = int(numSamples * percentTr + 0.5)
        Xtr = X_data[newInds[0:noTrain], :]
        Xtest = X_data[newInds[noTrain:], :]
    else:
        if percentTr == 1:
            Xtr = X_data
            Xtest = np.array([])
        else:
            Xtr = np.array([])
            Xtest = X_data
        
    return (Xtr, Xtest)


def saveDataToFile(path, X_data):
    """
    Save data to file
    
    INPUT
        path        The path to the data file (including file name)
        X_data      The data need to be stored
    """
    np.savetxt(path, X_data, fmt='%f', delimiter=', ')   
    

def string_to_boolean(st):
    if st == "True" or st == "true":
        return True
    elif st == "False" or st == "false":
        return False
    else:
        raise ValueError
        

def splitDatasetRndToKPart(Xl, Xu, patClassId, k = 10, isNorm = False, norm_range = [0, 1]):
    """
    Split a dataset into k parts randomly.
    
        INPUT
            Xl              Input data lower bounds (rows = objects, columns = features)
            X_u             Input data upper bounds (rows = objects, columns = features)
            patClassId      Input data class labels (crisp)
            k               Number of parts needs to be split
            isNorm          Do normalization of input training samples or not?
            norm_range      New ranging of input data after normalization, for example: [0, 1]
            
        OUTPUT
            partitionedA    An numpy array contains k sub-arrays, in which each subarray is Bunch datatype:
                                + lower:    lower bounds
                                + upper:    upper bounds
                                + label:    class labels
    """
    if isNorm == True:
        Xl = normalize(Xl, norm_range)
        Xu = normalize(Xu, norm_range)
    
    numSamples = Xl.shape[0]
    # generate random permutation
    pos = np.random.permutation(numSamples)
    
    # Bin the positions into numClassifier partitions
    anchors = np.round(np.linspace(0, numSamples, k + 1)).astype(np.int64)
    
    partitionedA = np.empty(k, dtype=Bunch)
    
    # divide the training set into k sub-datasets
    for i in range(k):
        partitionedA[i] = Bunch(lower = Xl[pos[anchors[i]:anchors[i + 1]], :], upper = Xu[pos[anchors[i]:anchors[i + 1]], :], label = patClassId[pos[anchors[i]:anchors[i + 1]]])
        
    return partitionedA
    
  
def splitDatasetRndClassBasedToKPart(Xl, Xu, patClassId, k= 10, isNorm = False, norm_range = [0, 1]):
    """
    Split a dataset into k parts randomly according to each class, where the number of samples of each class is equal among subsets
    
        INPUT
            Xl              Input data lower bounds (rows = objects, columns = features)
            X_u             Input data upper bounds (rows = objects, columns = features)
            patClassId      Input data class labels (crisp)
            k               Number of parts needs to be split
            isNorm          Do normalization of input training samples or not?
            norm_range      New ranging of input data after normalization, for example: [0, 1]
            
        OUTPUT
            partitionedA    An numpy array contains k sub-arrays, in which each subarray is Bunch datatype:
                                + lower:    lower bounds
                                + upper:    upper bounds
                                + label:    class labels
    """
    if isNorm == True:
        Xl = normalize(Xl, norm_range)
        Xu = normalize(Xu, norm_range)
        
    classes = np.unique(patClassId)
    partitionedA = np.empty(k, dtype=Bunch)
    
    for cl in range(classes.size):
        # Find indices of input samples having the same label with classes[cl]
        indClass = patClassId == classes[cl]
        # filter samples having the same class label with classes[cl]
        Xl_cl = Xl[indClass]
        Xu_cl = Xu[indClass]
        pathClass_cl = patClassId[indClass]
        
        numSamples = Xl_cl.shape[0]
        # generate random permutation of positions of selected patterns
        pos = np.random.permutation(numSamples)
        
        # Bin the positions into k partitions
        anchors = np.round(np.linspace(0, numSamples, k + 1)).astype(np.int64)
        
        for i in range(k):
            if cl == 0:
                lower_tmp = Xl_cl[pos[anchors[i]:anchors[i + 1]], :]
                upper_tmp = Xu_cl[pos[anchors[i]:anchors[i + 1]], :]
                label_tmp = pathClass_cl[pos[anchors[i]:anchors[i + 1]]]
                partitionedA[i] = Bunch(lower = lower_tmp, upper = upper_tmp, label = label_tmp)
            else:
                lower_tmp = np.concatenate((partitionedA[i].lower, Xl_cl[pos[anchors[i]:anchors[i + 1]], :]), axis=0)
                upper_tmp = np.concatenate((partitionedA[i].upper, Xu_cl[pos[anchors[i]:anchors[i + 1]], :]), axis=0)
                label_tmp = np.append(partitionedA[i].label, pathClass_cl[pos[anchors[i]:anchors[i + 1]]])
                partitionedA[i] = Bunch(lower = lower_tmp, upper = upper_tmp, label = label_tmp)
        
    return partitionedA


def splitDatasetRndClassBasedTo2Part(Xl, Xu, patClassId, training_rate = 0.5, isNorm = False, norm_range = [0, 1]):
    """
    Split a dataset into 2 parts randomly according to each class, the proposition training_rate is applied for each class
    
        INPUT
            Xl              Input data lower bounds (rows = objects, columns = features)
            X_u             Input data upper bounds (rows = objects, columns = features)
            patClassId      Input data class labels (crisp)
            training_rate   The percentage of the number of training samples needs to be split
            isNorm          Do normalization of input training samples or not?
            norm_range      New ranging of input data after normalization, for example: [0, 1]
            
        OUTPUT
            trainingSet     One object belonging to Bunch datatype contains training data with the following attributes:
                                + lower:    lower bounds
                                + upper:    upper bounds
                                + label:    class labels
            validSet        One object belonging to Bunch datatype contains validation data with the following attributes:
                                + lower:    lower bounds
                                + upper:    upper bounds
                                + label:    class labels
            
    """
    if isNorm == True:
        Xl = normalize(Xl, norm_range)
        Xu = normalize(Xu, norm_range)
        
    classes = np.unique(patClassId)
    trainingSet = None
    validSet = None
    
    for cl in range(classes.size):
        # Find indices of input samples having the same label with classes[cl]
        indClass = patClassId == classes[cl]
        # filter samples having the same class label with classes[cl]
        Xl_cl = Xl[indClass]
        Xu_cl = Xu[indClass]
        pathClass_cl = patClassId[indClass]
        
        numSamples = Xl_cl.shape[0]
        # generate random permutation of positions of selected patterns
        pos = np.random.permutation(numSamples)
        
        # Find the cut-off position
        pivot = int(numSamples * training_rate)
        
        if cl == 0:
            trainingSet = Bunch(lower = Xl_cl[pos[0:pivot]], upper = Xu_cl[pos[0:pivot]], label = pathClass_cl[pos[0:pivot]])
            validSet = Bunch(lower = Xl_cl[pos[pivot:]], upper = Xu_cl[pos[pivot:]], label = pathClass_cl[pos[pivot:]])
        else:
            lower_train = np.concatenate((trainingSet.lower, Xl_cl[pos[0:pivot]]), axis=0)
            upper_train = np.concatenate((trainingSet.upper, Xu_cl[pos[0:pivot]]), axis=0)
            label_train = np.append(trainingSet.label, pathClass_cl[pos[0:pivot]])
            trainingSet = Bunch(lower = lower_train, upper = upper_train, label = label_train)
            
            lower_valid = np.concatenate((validSet.lower, Xl_cl[pos[pivot:]]), axis=0)
            upper_valid = np.concatenate((validSet.upper, Xu_cl[pos[pivot:]]), axis=0)
            label_valid = np.append(validSet.label, pathClass_cl[pos[pivot:]])
            validSet = Bunch(lower = lower_valid, upper = upper_valid, label = label_valid)
            
        
    return (trainingSet, validSet)


def splitDatasetRndTo2Part(Xl, Xu, patClassId, training_rate = 0.5, isNorm = False, norm_range = [0, 1]):
    """
    Split a dataset into 2 parts randomly on whole dataset, the proposition training_rate is applied for whole dataset
    
        INPUT
            Xl              Input data lower bounds (rows = objects, columns = features)
            X_u             Input data upper bounds (rows = objects, columns = features)
            patClassId      Input data class labels (crisp)
            training_rate   The percentage of the number of training samples needs to be split
            isNorm          Do normalization of input training samples or not?
            norm_range      New ranging of input data after normalization, for example: [0, 1]
            
        OUTPUT
            trainingSet     One object belonging to Bunch datatype contains training data with the following attributes:
                                + lower:    lower bounds
                                + upper:    upper bounds
                                + label:    class labels
            validSet        One object belonging to Bunch datatype contains validation data with the following attributes:
                                + lower:    lower bounds
                                + upper:    upper bounds
                                + label:    class labels
            
    """
    if isNorm == True:
        Xl = normalize(Xl, norm_range)
        Xu = normalize(Xu, norm_range)
    
    numSamples = Xl.shape[0]
    # generate random permutation
    pos = np.random.permutation(numSamples)
    
    # Find the cut-off position
    pivot = int(numSamples * training_rate)
    
    # divide the training set into 2 sub-datasets
    trainingSet = Bunch(lower = Xl[pos[0:pivot]], upper = Xu[pos[0:pivot]], label = patClassId[pos[0:pivot]])
    validSet = Bunch(lower = Xl[pos[pivot:]], upper = Xu[pos[pivot:]], label = patClassId[pos[pivot:]])
    
    return (trainingSet, validSet)