# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 22:26:26 2018

@author: thanh
"""

import sys, os
sys.path.insert(0, os.path.pardir) 
#import os,sys,inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0, parentdir)

import ast
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')

from functionhelper.membershipcalc import memberG
from functionhelper.hyperboxadjustment import hyperboxOverlapTest, hyperboxContraction
from GFMM.classification import predict
from functionhelper.drawinghelper import drawbox
from functionhelper.prepocessinghelper import loadDataset, string_to_boolean
from GFMM.basegfmmclassifier import BaseGFMMClassifier

class OnlineGFMM(BaseGFMMClassifier):
    
    def __init__(self, gamma = 1, teta = 1, tMin = 1, isDraw = False, oper = 'min', isNorm = False, norm_range = [0, 1], V = np.array([], dtype=np.float64), W = np.array([], dtype=np.float64), classId = np.array([], dtype=np.int16)):
        BaseGFMMClassifier.__init__(self, gamma, teta, isDraw, oper, isNorm, norm_range)
        
        self.tMin = tMin
        self.V = V
        self.W = W
        self.classId = classId
        self.misclass = 1
        self.cardin = list()
        
    
    def calculateProbability(self, idHyperbox, X_l, X_u, memVal):
        """
        Compute the selected probability of current hyperbox
        
        INPUT:
            + idHyperbox        Index of the hyperbox being considered
            + X_l, X_u          Lower and upper bounds of input data
            
        OUTPUT:
            The probability value = the number of samples located in hyperbox / total samples belonging to the hyperbox
        """
        index_Samples = self.cardin[idHyperbox]
        num_in = num_out = 0
        
        for i in index_Samples:
            b = memberG(X_l[i], X_u[i], self.V[idHyperbox], self.W[idHyperbox], self.gamma)
            
            if b[0] == 1:
                num_in = num_in + 1  # Increate the number of samples located within the current hyperbox
            else:
                num_out = num_out + 1
        
        if num_in + num_out == 0:
            prob = 1
        else:
            # prob = (3 * (num_in / (num_in + num_out)) + memVal) / 4
            prob = num_in / (num_in + num_out)
        
        return prob
    
        
    def fit(self, X_l, X_u, patClassId):
        """
        Training the classifier
        
         Xl             Input data lower bounds (rows = objects, columns = features)
         Xu             Input data upper bounds (rows = objects, columns = features)
         patClassId     Input data class labels (crisp). patClassId[i] = 0 corresponds to an unlabeled item
        
        """
        print('--Probability Online Learning--')
        
        if self.isNorm == True:
            X_l, X_u = self.dataPreprocessing(X_l, X_u)
            
        time_start = time.clock()
        
        yX, xX = X_l.shape
        teta = self.teta
        
        mark = np.array(['*', 'o', 'x', '+', '.', ',', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H', 'X', 'D', '|', '_'])
        mark_col = np.array(['r', 'g', 'b', 'y', 'c', 'm', 'k'])
        
        listLines = list()
        listInputSamplePoints = list();
        
        if self.isDraw:
            drawing_canvas = self.initializeCanvasGraph("GFMM - Probability Online learning", xX)
            if self.V.size > 0:
                # draw existed hyperboxes
                color_ = np.array(['k'] * len(self.classId), dtype = object)
                for c in range(len(self.classId)):
                    if self.classId[c] < len(mark_col):
                        color_[c] = mark_col[self.classId[c]]
                
                hyperboxes = drawbox(self.V[:, 0:np.minimum(xX,3)], self.W[:, 0:np.minimum(xX,3)], drawing_canvas, color_)
                listLines.extend(hyperboxes)
                self.delay()
            
        while self.misclass > 0 and teta >= self.tMin:
            # for each input sample
            for j in range(len(self.cardin)):
                self.cardin[j] = np.array([], dtype = np.int64)
                
            for i in range(yX):
                classOfX = patClassId[i]
                # draw input samples
                if self.isDraw:
                    if i == 0 and len(listInputSamplePoints) > 0:
                        # reset input point drawing
                        for point in listInputSamplePoints:
                            point.remove()
                        listInputSamplePoints.clear()
                    
                    color_ = 'k'
                    if classOfX < len(mark_col):
                        color_ = mark_col[classOfX]
                    
                    if (X_l[i, :] == X_u[i, :]).all():
                        marker_ = 'd'                   
                        if classOfX < len(mark):
                            marker_ = mark[classOfX]
                            
                        if xX == 2:
                            inputPoint = drawing_canvas.plot(X_l[i, 0], X_l[i, 1], color = color_, marker=marker_)
                        else:
                            inputPoint = drawing_canvas.plot([X_l[i, 0]], [X_l[i, 1]], [X_l[i, 2]], color = color_, marker=marker_)
                        
                        #listInputSamplePoints.append(inputPoint)
                    else:
                        inputPoint = drawbox(np.asmatrix(X_l[i, 0:np.minimum(xX, 3)]), np.asmatrix(X_u[i, 0:np.minimum(xX, 3)]), drawing_canvas, color_)
                        
                    listInputSamplePoints.append(inputPoint[0])
                    self.delay()
                    
                if self.V.size == 0:   # no model provided - starting from scratch
                    self.V = np.array([X_l[0]])
                    self.W = np.array([X_u[0]])
                    self.classId = np.array([patClassId[0]])
                    self.cardin.append(np.array([0], dtype=np.int64))
                    
                    if self.isDraw == True:
                        # draw hyperbox
                        box_color = 'k'
                        if patClassId[0] < len(mark_col):
                            box_color = mark_col[patClassId[0]]
                        
                        hyperbox = drawbox(np.asmatrix(self.V[0, 0:np.minimum(xX,3)]), np.asmatrix(self.W[0, 0:np.minimum(xX,3)]), drawing_canvas, box_color)
                        listLines.append(hyperbox[0])
                        self.delay()

                else:
                    b = memberG(X_l[i], X_u[i], self.V, self.W, self.gamma)
                        
                    index = np.argsort(b)[::-1]
                    bSort = b[index];
                    
                    if bSort[0] != 1 or (classOfX != self.classId[index[0]] and classOfX != 0):
                        adjust = False

                        for j in index:
                            if classOfX == self.classId[j] or self.classId[j] == 0 or classOfX == 0:
#                                zz = zz + 1
#                                if zz == 10:
#                                    break
                                # test violation of max hyperbox size and class labels
#                                selected_Prob = self.calculateProbability(j, X_l, X_u, bSort[j])
#                                print('selected_Prob =', selected_Prob)
                                # if np.random.rand() <= selected_Prob and ((np.maximum(self.W[j], X_u[i]) - np.minimum(self.V[j], X_l[i])) <= teta).all() == True:
                                    # adjust the j-th hyperbox
                                if ((np.maximum(self.W[j], X_u[i]) - np.minimum(self.V[j], X_l[i])) <= teta).all() == True:
                                    selected_Prob = self.calculateProbability(j, X_l, X_u, bSort[j])
                                    
                                    if np.random.rand() <= selected_Prob:
                                        self.V[j] = np.minimum(self.V[j], X_l[i])
                                        self.W[j] = np.maximum(self.W[j], X_u[i])
                                        
                                        self.cardin[j] = np.append(self.cardin[j], i)
                                        
                                        indOfWinner = j
                                        adjust = True
                                        if classOfX != 0 and self.classId[j] == 0:
                                            self.classId[j] = classOfX
                                        
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
                                            
                                        break
                                
                        # if i-th sample did not fit into any existing box, create a new one
                        if not adjust:
                            self.V = np.vstack((self.V, X_l[i]))
                            self.W = np.vstack((self.W, X_u[i]))
                            self.classId = np.append(self.classId, classOfX)
                            self.cardin.append(np.array([i], dtype=np.int64))

                            if self.isDraw:
                                # handle drawing graph
                                box_color = 'k'
                                if self.classId[-1] < len(mark_col):
                                    box_color = mark_col[self.classId[-1]]
                                    
                                hyperbox = drawbox(np.asmatrix(X_l[i, 0:np.minimum(xX, 3)]), np.asmatrix(X_u[i, 0:np.minimum(xX, 3)]), drawing_canvas, box_color)
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
                        self.cardin[index[0]] = np.append(self.cardin[index[0]], i)
                       
            teta = teta * 0.9
            result = predict(self.V, self.W, self.classId, X_l, X_u, patClassId, self.gamma, self.oper)
            self.misclass = result.summis

        # Draw last result  
#        if self.isDraw == True:
#            # Handle drawing graph
#            drawing_canvas.cla()
#            color_ = np.empty(len(self.classId), dtype = object)
#            for c in range(len(self.classId)):
#                color_[c] = mark_col[self.classId[c]]
#                
#            drawbox(self.V[:, 0:np.minimum(xX, 3)], self.W[:, 0:np.minimum(xX, 3)], drawing_canvas, color_)
#            self.delay()
#                
#        if self.isDraw:
#            plt.show()
        
        time_end = time.clock()
        self.elapsed_training_time = time_end - time_start
        
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
    arg6: + The minimum value of maximum size of hyperboxes (teta_min: default = teta)
    arg7: + gamma value (default: 1)
    arg8: operation used to compute membership value: 'min' or 'prod' (default: 'min')
    arg9: + do normalization of datasets or not? True: Normilize, False: No normalize (default: True)
    arg10: + range of input values after normalization (default: [0, 1])   
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
        teta_min = teta
    else:
        teta_min = float(sys.argv[6])
    
    if len(sys.argv) < 8:
        gamma = 1
    else:
        gamma = float(sys.argv[7])
    
    if len(sys.argv) < 9:
        oper = 'min'
    else:
        oper = sys.argv[8]
    
    if len(sys.argv) < 10:
        isNorm = True
    else:
        isNorm = string_to_boolean(sys.argv[9])
    
    if len(sys.argv) < 11:
        norm_range = [0, 1]
    else:
        norm_range = ast.literal_eval(sys.argv[10])
    
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
        
    classifier = OnlineGFMM(gamma, teta, teta_min, isDraw, oper, isNorm, norm_range)
    classifier.fit(Xtr, Xtr, patClassIdTr)
    
    # Testing
    print("-- Testing --")
    result = classifier.predict(Xtest, Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")