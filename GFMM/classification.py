# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:22:08 2018

@author: Thanh Tung Khuat

GFMM Predictor

"""

import numpy as np
from membershipcalc import memberG
from bunchdatatype import Bunch

def predict(V, W, classId, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine)
    
      result = predict(V,W,classId,XlT,XuT,patClassIdTest,gama,oper)
  
    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	          Input data (hyperbox) class labels (crisp)
      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')
  
   OUTPUT
      result           A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + sumamb           Number of objects with maximum membership in more than one class
                          + out              Soft class memberships
                          + mem              Hyperbox memberships

    """

    #initialization
    yX = XlT.shape[0]
    misclass = np.zeros(yX)
    classes = np.unique(np.sort(classId))
    noClasses = classes.size
    ambiguity = np.zeros((yX, 1))
    mem = np.zeros((yX, V.shape[0]))
    out = np.zeros((yX, noClasses))

    # classifications
    for i in range(yX):
        mem[i, :] = memberG(XlT[i, :], XuT[i, :], np.minimum(V, W), np.maximum(W, V), gama, oper) # calculate memberships for all hyperboxes
        bmax = mem[i,:].max()	                                          # get max membership value
        maxVind = np.nonzero(mem[i,:] == bmax)[0]                         # get indexes of all hyperboxes with max membership
        
        for j in range(noClasses):
            out[i, j] = mem[i, classId == classes[j]].max()            # get max memberships for each class
        
        ambiguity[i, :] = np.sum(out[i, :] == bmax) 						  # number of different classes with max membership
        
        if bmax == 0:
            print('zero maximum membership value')                     # this is probably bad...
            
        misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))
    
    # results
    sumamb = np.sum(ambiguity[:, 0] > 1)
    summis = np.sum(misclass).astype(np.int64)
    
    result = Bunch(summis = summis, misclass = misclass, sumamb = sumamb, out = out, mem = mem)
    return result


def predictDecisionLevelEnsemble(classifiers, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    Perform classification for a decision level ensemble learning
    
                result = predictDecisionLevelEnsemble(classifiers, XlT, XuT, patClassIdTest, gama, oper)
    
    INPUT
        classifiers         An array of classifiers needed to combine, datatype of each element in the array is BaseGFMMClassifier
        XlT                 Test data lower bounds (rows = objects, columns = features)
        XuT                 Test data upper bounds (rows = objects, columns = features)
        patClassIdTest      Test data class labels (crisp)
        gama                Membership function slope (default: 1)
        oper                Membership calculation operation: 'min' or 'prod' (default: 'min')
        
    OUTPUT
        result              A object with Bunch datatype containing all results as follows:
                                + summis        Number of misclassified samples
                                + misclass      Binary error map for input samples
                                + out           Soft class memberships, rows are testing input patterns, columns are indices of classes
                                + classes       Store class labels corresponding column indices of out
    """
    numClassifier = len(classifiers)
    
    yX = XlT.shape[0]
    misclass = np.zeros(yX, dtype=np.bool)
    classes = np.unique(classifiers[0].classId)
    noClasses = len(classes)
    out = np.zeros((yX, noClasses), dtype=np.float64)
    
    # classification of each testing pattern i
    for i in range(yX):
        for idClf in range(numClassifier):
            # calculate memberships for all hyperboxes of classifier idClf
            mem_tmp = memberG(XlT[i, :], XuT[i, :], classifiers[idClf].V, classifiers[idClf].W, gama, oper)
            
            for j in range(noClasses):
                # get max membership of hyperobxes with class label j
                mem_max = mem_tmp[classifiers[idClf].classId == classes[j]].max()
                
                if len(mem_max) > 0:
                    out[i, j] = out[i, j] + mem_max
        
        # compute membership value of each class over all classifiers            
        out[i, :] = out[i, :] / numClassifier
        # get max membership value for each class with regard to the i-th sample
        maxb = out[i].max()
        # get positions of indices of all classes with max membership
        maxMemInd = out[i] == maxb
        misclass[i] = np.logical_or((classes[maxMemInd] == patClassIdTest[i]).any(), patClassIdTest[i] == 0)
        
    # count number of missclassified patterns
    summis = np.sum(misclass)
    
    result = Bunch(summis = summis, misclass = misclass, out = out, classes = classes)
    return result
        
        
    