# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 18:23:03 2018

@author: Thanh Tung Khuat

Hyperbox adjustment handling: overlap testing, hyperbox contraction

"""
import numpy as np

def hyperboxOverlapTest(V, W, ind, testInd):
    """
    Hyperbox overlap test

      dim = overlapTest(V, W, ind, testInd)
  
    INPUT
        V           Hyperbox lower bounds
        W           Hyperbox upper bounds
        ind         Index of extended hyperbox
        testInd     Index of hyperbox to test for overlap with the extended hyperbox

    OUTPUT
        dim         Result to be fed into contrG1, which is special numpy array

    """
    dim = np.array([]);
    xW = W.shape[1]
    
    condWiWk = W[ind, :] - W[testInd, :] > 0
    condViVk = V[ind, :] - V[testInd, :] > 0
    condWkVi = W[testInd, :] - V[ind, :] > 0
    condWiVk = W[ind, :] - V[testInd, :] > 0

    c1 = ~condWiWk & ~condViVk & condWiVk
    c2 = condWiWk & condViVk & condWkVi;
    c3 = condWiWk & ~condViVk;
    c4 = ~condWiWk & condViVk
    c = c1 + c2 + c3 + c4

    ad = c.all()

    if ad == True:
        minimum = 1;
        for i in range(xW):
            if c1[i] == True:
                if minimum > W[ind, i] - V[testInd, i]:
                    minimum = W[ind, i] - V[testInd, i]
                    dim = np.array([1, i])
            
            elif c2[i] == True:
                if minimum > W[testInd, i] - V[ind, i]:
                    minimum = W[testInd, i] - V[ind, i]
                    dim = np.array([2, i])
            
            elif c3[i] == True:
                if minimum > (W[testInd, i] - V[ind,i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                    dim = np.array([31, i])
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                    dim = np.array([32, i])
                    
            elif c4[i] == True:
                if minimum > (W[testInd, i] - V[ind, i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                    dim = np.array([41, i])
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                    dim = np.array([42, i])
                
    return dim

def hyperboxContraction(V1, W1, newCD, testedInd, ind):
    """
    Adjusting min-max points of overlaping clusters (with meet halfway)

      V, W = hyperboxContraction(V,W,newCD,testedInd,ind)
  
    INPUT
      V1            Lower bounds of existing hyperboxes
      W1            Upper bounds of existing hyperboxes
      newCD         Special parameters, output from overlapTest
      testedInd     Index of hyperbox to test for overlap with the extended hyperbox
      ind           Index of extended hyperbox	
   
    OUTPUT
      V             Lower bounds of adjusted hyperboxes
      W             Upper bounds of adjusted hyperboxes
    
    """
    V = V1.copy()
    W = W1.copy()
    if newCD[0] == 1:
        W[ind, newCD[1]] = (V[testedInd, newCD[1]] + W[ind, newCD[1]]) / 2
        V[testedInd, newCD[1]] = W[ind, newCD[1]]
    elif newCD[0] == 2:
        V[ind, newCD[1]] = (W[testedInd, newCD[1]] + V[ind, newCD[1]]) / 2
        W[testedInd, newCD[1]] = V[ind, newCD[1]]
    elif newCD[0] == 31:
        V[ind, newCD[1]] = W[testedInd, newCD[1]]
    elif newCD[0] == 32:
        W[ind, newCD[1]] = V[testedInd, newCD[1]]
    elif newCD[0] == 41:
        W[testedInd, newCD[1]] = V[ind, newCD[1]]
    elif newCD[0] == 42:
        V[testedInd, newCD[1]] = W[ind, newCD[1]]
    
    return (V, W)


def isOverlap(V, W, ind, classId):
    """
    Checking overlap between hyperbox ind and remaning hyperboxes (1 vs many)
    
    INPUT
        V           Hyperbox lower bounds
        W           Hyperbox upper bounds
        ind         Index of the hyperbox to be checked for overlap
        classId     Class labels of hyperboxes
        
    OUTPUT
        False - no overlap,  True - overlap
    """
    
    if (V[ind] > W[ind]).any() == True:
        return False
    else:
        indcomp = np.nonzero((W >= V).all(axis = 1))[0] 	# examine only hyperboxes w/o missing dimensions, meaning that in each dimension upper bound is larger than lowerbound
        
        if len(indcomp) == 0:
            return False
        else:
            testedHyperIndex = np.where(indcomp == ind)[0][0]
            # remove index of the tested hyperbox
            #newInd = np.delete(indcomp, np.where(indcomp == ind)[0])
            newInd = np.append(indcomp[0:testedHyperIndex], indcomp[testedHyperIndex:])
            
            if len(newInd) > 0:
                onesTemp = np.ones((len(newInd), 1))
                condWiWk = onesTemp * W[ind] - W[newInd] > 0
                condViVk = onesTemp * V[ind] - V[newInd] > 0
                condWkVi = W[newInd] - onesTemp * V[ind] > 0
                condWiVk = onesTemp * W[ind] - V[newInd] > 0
                
                c1 = ~condWiWk & ~condViVk & condWiVk
                c2 = condWiWk & condViVk & condWkVi
                c3 = condWiWk & ~condViVk
                c4 = ~condWiWk & condViVk
                
                c = c1 + c2 + c3 + c4
                ad = c.all(axis = 1)
                ind2 = newInd[ad]
                
                ovresult = (classId[ind2] != classId[ind]).any()
                
                return ovresult
            else:
                return False