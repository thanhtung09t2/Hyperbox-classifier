# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 14:00:25 2018

@author: Thanh Tung Khuat

Implementation of the original fuzzy min-max neural network

            FMNNClassification(gamma, teta, isDraw, isNorm, norm_range)

    INPUT
     V              Hyperbox lower bounds for the model to be updated using new data
     W              Hyperbox upper bounds for the model to be updated using new data
     classId        Hyperbox class labels (crisp)  for the model to be updated using new data
     gamma          Membership function slope (default: 1), datatype: array or scalar
     teta           Maximum hyperbox size (default: 1)
     isDraw         Progress plot flag (default: False)
     isNorm         Do normalization of input training samples or not?
     norm_range     New ranging of input data after normalization

"""
import sys, os
sys.path.insert(0, os.path.pardir) 

import ast
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from functionhelper.membershipcalc import simpsonMembership
from functionhelper.hyperboxadjustment import hyperboxOverlapTest, hyperboxContraction
from functionhelper.drawinghelper import drawbox
from functionhelper.prepocessinghelper import loadDataset, string_to_boolean
from functionhelper.basefmnnclassifier import BaseFMNNClassifier

class FMNNClassification(BaseFMNNClassifier):
    
    def __init__(self, gamma = 1, teta = 1, isDraw = False, isNorm = False, norm_range = [0, 1], V = np.array([], dtype=np.float64), W = np.array([], dtype=np.float64), classId = np.array([], dtype=np.int16)):
        BaseFMNNClassifier.__init__(self, gamma, teta, isDraw, isNorm, norm_range)
        
        self.V = V
        self.W = W
        self.classId = classId
        
    
    def fit(self, Xh, patClassId):
        """
        Training the classifier
        
         Xh             Input data (rows = objects, columns = features)
         patClassId     Input data class labels (crisp). patClassId[i] = 0 corresponds to an unlabeled item
        
        """
        print('--Online Learning for Simpson''s FMNN--')
        
        if self.isNorm == True:
            Xh = self.dataPreprocessing(Xh)
        
        yX, xX = Xh.shape
        
        mark = np.array(['*', 'o', 'x', '+', '.', ',', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H', 'X', 'D', '|', '_'])
        mark_col = np.array(['r', 'g', 'b', 'y', 'c', 'm', 'k'])
        
        listLines = list()
        
        if self.isDraw:
            drawing_canvas = self.initializeCanvasGraph("FMNN - Simpson's fuzzy min-max neural network", xX)
            
        # for each input sample
        for i in range(yX):
            classOfX = patClassId[i]
            # draw input samples
            if self.isDraw:
                
                color_ = 'k'
                if classOfX < len(mark_col):
                    color_ = mark_col[classOfX]
                
                marker_ = 'd'                   
                if classOfX < len(mark):
                    marker_ = mark[classOfX]
                    
                if xX == 2:
                    drawing_canvas.plot(Xh[i, 0], Xh[i, 1], color = color_, marker=marker_)
                else:
                    drawing_canvas.plot(Xh[i, 0], Xh[i, 1], Xh[i, 2], color = color_, marker=marker_)
                
                self.delay()
                
            if self.V.size == 0:   # no model provided - starting from scratch
                self.V = np.array([Xh[0]])
                self.W = np.array([Xh[0]])
                self.classId = np.array([patClassId[0]])
                
                if self.isDraw == True:
                    # draw hyperbox
                    box_color = 'k'
                    if patClassId[0] < len(mark_col):
                        box_color = mark_col[patClassId[0]]
                    
                    hyperbox = drawbox(np.asmatrix(self.V[0, 0:np.minimum(xX,3)]), np.asmatrix(self.W[0, 0:np.minimum(xX,3)]), drawing_canvas, box_color)
                    listLines.append(hyperbox[0])
                    self.delay()

            else:
                idSameClassOfX = np.where(self.classId == classOfX)[0]
                # Find all hyperboxes same class with indexOfX
                V1 = self.V[idSameClassOfX]
                
                if len(V1) > 0:
                    W1 = self.W[idSameClassOfX]
                    
                    b = simpsonMembership(Xh[i], V1, W1, self.gamma)
                    
                    indexSort = np.argsort(b)[::-1]
                    
                    # store the index of the winner hyperbox in the list of all hyperboxes of all classes
                    j = idSameClassOfX[indexSort[0]]
                
                    if b[indexSort[0]] != 1:
                        adjust = False
                
                        # test violation of max hyperbox size and class labels
                        if (np.maximum(self.W[j], Xh[i]) - np.minimum(self.V[j], Xh[i])).sum() <= self.teta * xX:
                            # adjust the j-th hyperbox
                            self.V[j] = np.minimum(self.V[j], Xh[i])
                            self.W[j] = np.maximum(self.W[j], Xh[i])
                            indOfWinner = j
                            adjust = True
                            
                            if self.isDraw:
                                # Handle drawing graph
                                box_color = 'k'
                                if self.classId[j] < len(mark_col):
                                    box_color = mark_col[self.classId[j]]
                                
                                try:
                                    listLines[j].remove()
                                except:
                                    pass
                                
                                hyperbox = drawbox(np.asmatrix(self.V[j, 0:np.minimum(xX, 3)]), np.asmatrix(self.W[j, 0:np.minimum(xX, 3)]), drawing_canvas, box_color)                                 
                                listLines[j] = hyperbox[0]
                                self.delay()
                                    
                                
                        # if i-th sample did not fit into any existing box, create a new one
                        if not adjust:
                            self.V = np.vstack((self.V, Xh[i]))
                            self.W = np.vstack((self.W, Xh[i]))
                            self.classId = np.append(self.classId, classOfX)
    
                            if self.isDraw:
                                # handle drawing graph
                                box_color = 'k'
                                if self.classId[-1] < len(mark_col):
                                    box_color = mark_col[self.classId[-1]]
                                    
                                hyperbox = drawbox(np.asmatrix(Xh[i, 0:np.minimum(xX, 3)]), np.asmatrix(Xh[i, 0:np.minimum(xX, 3)]), drawing_canvas, box_color)
                                listLines.append(hyperbox[0])
                                self.delay()
                                
                        elif self.V.shape[0] > 1:
                            for ii in range(self.V.shape[0]):
                                if ii != indOfWinner:
                                    caseDim = hyperboxOverlapTest(self.V, self.W, indOfWinner, ii)		# overlap test
                                    
                                    if caseDim.size > 0 and self.classId[ii] != self.classId[indOfWinner]:
                                        self.V, self.W = hyperboxContraction(self.V, self.W, caseDim, ii, indOfWinner)
                                        if self.isDraw:
                                            # Handle graph drawing
                                            boxii_color = boxwin_color = 'k'
                                            if self.classId[ii] < len(mark_col):
                                                boxii_color = mark_col[self.classId[ii]]
                                            
                                            if self.classId[indOfWinner] < len(mark_col):
                                                boxwin_color = mark_col[self.classId[indOfWinner]]
                                            
                                            try:
                                                listLines[ii].remove()                                           
                                                listLines[indOfWinner].remove()
                                            except:
                                                pass
                                            
                                            hyperboxes = drawbox(self.V[[ii, indOfWinner], 0:np.minimum(xX, 3)], self.W[[ii, indOfWinner], 0:np.minimum(xX, 3)], drawing_canvas, [boxii_color, boxwin_color])                                          
                                            listLines[ii] = hyperboxes[0]
                                            listLines[indOfWinner] = hyperboxes[1]                                      
                                            self.delay()
                                            
                else:
                    # create a new hyperbox
                    self.V = np.vstack((self.V, Xh[i]))
                    self.W = np.vstack((self.W, Xh[i]))
                    self.classId = np.append(self.classId, classOfX)
    
                    if self.isDraw:
                        # handle drawing graph
                        box_color = 'k'
                        if self.classId[-1] < len(mark_col):
                            box_color = mark_col[self.classId[-1]]
                            
                            hyperbox = drawbox(np.asmatrix(Xh[i, 0:np.minimum(xX, 3)]), np.asmatrix(Xh[i, 0:np.minimum(xX, 3)]), drawing_canvas, box_color)
                            listLines.append(hyperbox[0])
                            self.delay()
                            
           						
        return self
    
    
if __name__ == '__main__':
    """
    INPUT parameters from command line
    arg1: + 1 - training and testing datasets are located in separated files
          + 2 - training and testing datasets are located in the same files
    arg2: path to file containing the training dataset (arg1 = 1) or both training and testing datasets (arg1 = 2)
    arg3: + path to file containing the testing dataset (arg1 = 1)
          + percentage of the training dataset in the input file
    arg4: + True: drawing hyperboxes during the training process
          + False: no drawing
    arg5: + Maximum size of hyperboxes (teta, default: 1)
    arg6: + gamma value (default: 1)
    arg7: + do normalization of datasets or not? True: Normilize, False: No normalize (default: True)
    arg8: + range of input values after normalization (default: [0, 1])   
    """
    # Init default parameters
    if len(sys.argv) < 5:
        isDraw = False
    else:
        isDraw = string_to_boolean(sys.argv[4])
    
    if len(sys.argv) < 6:
        teta = 1    
    else:
        teta = float(sys.argv[5])
    
    if len(sys.argv) < 7:
        gamma = 1
    else:
        gamma = float(sys.argv[6])

    if len(sys.argv) < 8:
        isNorm = True
    else:
        isNorm = string_to_boolean(sys.argv[7])
    
    if len(sys.argv) < 9:
        norm_range = [0, 1]
    else:
        norm_range = ast.literal_eval(sys.argv[8])
    
    # print('isDraw = ', isDraw, ' teta = ', teta, ' teta_min = ', teta_min, ' gamma = ', gamma, ' oper = ', oper, ' isNorm = ', isNorm, ' norm_range = ', norm_range)
    
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
        
    classifier = FMNNClassification(gamma, teta, isDraw, isNorm, norm_range)
    classifier.fit(Xtr, patClassIdTr)
    
    # Testing
    print("-- Testing --")
    result = classifier.predict(Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")