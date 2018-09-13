# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 14:57:55 2018

@author: Khuat Thanh Tung

Batch GFMM classifier (training core) - Faster version by only computing similarity among hyperboxes with the same label

    BatchGFMM(gama, teta, bthres, simil, sing, isDraw, oper, isNorm, norm_range, cardin, clusters)
  
    INPUT
        gama        Membership function slope (default: 1)
        teta        Maximum hyperbox size (default: 1)
        bthres		Similarity threshold for hyperbox concatenetion (default: 0.5)
        simil       Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
        sing        Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
        oper        Membership calculation operation: 'min' or 'prod' (default: 'min')
        isDraw      Progress plot flag (default: 1)
        oper        Membership calculation operation: 'min' or 'prod' (default: 'min')
        isNorm      Do normalization of input training samples or not?
        norm_range  New ranging of input data after normalization, for example: [0, 1]
        cardin      Input hyperbox cardinalities
        clusters    Identifiers of objects in each input hyperbox 
        
    ATTRIBUTES:
        V               Hyperbox lower bounds
        W               Hyperbox upper bounds
        classId         Hyperbox class labels (crisp)
        cardin          Hyperbox cardinalities (the number of training samples is covered by corresponding hyperboxes)
        clusters        Identifiers of input objects in each hyperbox (indexes of training samples covered by corresponding hyperboxes)

"""

def fit():
    """
    Xl          Input data lower bounds (rows = objects, columns = features)
    Xu          Input data upper bounds (rows = objects, columns = features)
    patClassId  Input data class labels (crisp)
    """