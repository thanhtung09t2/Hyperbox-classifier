# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:18:22 2018

@author: Thanh Tung Khuat

Preprocessing functions helper

"""

import numpy as np

def normalize(A, new_range):
    """
    Normalize the input dataset
    
    INPUT
        A           Original dataset (numpy array) [rows are samples, cols are features]
        new_range   The range of data after normalizing
   
    OUTPUT
        Normalized dataset
    """
    D = A.copy().astype(np.float_)
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
    A = np.array([], dtype=np.float_)
    with open(path) as f:
        for line in f:
            nums = np.fromstring(line.rstrip('\n').replace(',', ' '), dtype=np.float_, sep=' ')
            if (A.size == 0):
                A = nums
            else:
                A = np.vstack((A, nums))
    
    YA, XA = A.shape
   
    X_data = A[:, 0:XA-1]
    classId_dat = A[:, -1];
    classLabels = np.unique(classId_dat)
    
    # class labels must start from 1, class label = 0 means no label
    if classLabels.size > 1 and np.size(np.nonzero(classId_dat == 0)) > 0:
        classId_dat = classId_dat + 1

    if isNorm:
        X_data = normalize(X_data, new_range)
    
    Xtr = np.empty((0, XA - 1), dtype=np.float64)
    Xtest = np.empty((0, XA - 1), dtype=np.float64)
    
    patClassIdTr = np.array([], dtype=np.int64)
    patClassIdTest = np.array([], dtype=np.int64)
    
    if percentTr != 1 and percentTr != 0:
        noClasses = classLabels.size
        
        for k in range(noClasses):
            idx = np.nonzero(classId_dat == classLabels[k])[0]
            # randomly shuffle indices of elements belonging to class classLabels[k]
            if percentTr != 1 and percentTr != 0:
                idx = idx[np.random.permutation(len(idx))] 
    
            noTrain = int(len(idx) * percentTr + 0.5)
    
            # Attach data of class k to corresponding datasets
            Xtr_tmp = X_data[idx[0:noTrain], :]
            Xtr = np.vstack((Xtr, Xtr_tmp))
            patClassId_tmp = np.full(noTrain, classLabels[k], dtype=np.int16)
            patClassIdTr = np.append(patClassIdTr, patClassId_tmp)
            
            patClassId_tmp = np.full(len(idx) - noTrain, classLabels[k], dtype=np.int16)
            Xtest = np.vstack((Xtest, X_data[idx[noTrain:len(idx)], :]))
            patClassIdTest = np.append(patClassIdTest, patClassId_tmp)
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

def string_to_boolean(st):
    if st == "True" or st == "true":
        return True
    elif st == "False" or st == "false":
        return False
    else:
        raise ValueError