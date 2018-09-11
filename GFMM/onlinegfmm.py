# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 22:39:47 2018

@author: Thanh Tung Khuat

onlnGFMM - Online GFMM classifier (training core)

     OnlineGFMM(gamma, teta, tMin, isDraw, oper, V, W, classId, isNorm, norm_range)
  
   INPUT
     V              Hyperbox lower bounds for the model to be updated using new data
     W              Hyperbox upper bounds for the model to be updated using new data
     classId        Hyperbox class labels (crisp)  for the model to be updated using new data
     gamma          Membership function slope (default: 1), datatype: array or scalar
     teta           Maximum hyperbox size (default: 1)
     tMin           Minimum value of Teta
     isDraw         Progress plot flag (default: False)
     oper           Membership calculation operation: 'min' or 'prod' (default: 'min')
     isNorm         Do normalization of input training samples or not?
     norm_range     New ranging of input data after normalization    
"""
import sys
import ast
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from membershipcalc import memberG
from hyperboxadjustment import hyperboxOverlapTest, hyperboxContraction
from classification import predict
from drawinghelper import drawbox
from prepocessinghelper import loadDataset
from prepocessinghelper import normalize

class OnlineGFMM(object):
    
    def __init__(self, gamma = 1, teta = 1, tMin = 1, isDraw = False, oper = 'min', isNorm = False, norm_range = [0, 1], V = np.array([], dtype=np.float64), W = np.array([], dtype=np.float64), classId = np.array([], dtype=np.int16)):
        self.gamma = gamma
        self.teta = teta
        self.tMin = tMin
        self.V = V
        self.W = W
        self.classId = classId
        self.isDraw = isDraw
        self.oper = oper
        self.misclass = 1
        self.isNorm = isNorm
        self.loLim = norm_range[0]
        self.hiLim = norm_range[1]
        self.delayConstant = 0.001 # delay time period to display hyperboxes on the canvas
        
    def fit(self, X_l, X_u, patClassId):
        """
        Training the classifier
        
         Xl             Input data lower bounds (rows = objects, columns = features)
         Xu             Input data upper bounds (rows = objects, columns = features)
         patClassId     Input data class labels (crisp). patClassId[i] = 0 corresponds to an unlabeled item
        
        """
        print('--Online Learning--')
        yX, xX = X_l.shape
        teta = self.teta
        
        mark = np.array(['*', 'o', 'x', '+', '.', ',', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H', 'X', 'D', '|', '_'])
        mark_col = np.array(['r', 'g', 'b', 'y', 'c', 'm', 'k'])
        
        # Normalize input samples if needed
        if X_l.min() < self.loLim or X_u.min() < self.loLim or X_u.max() > self.hiLim or X_l.max() > self.hiLim:
            self.mins = X_l.min(axis = 0) # get min value of each feature
            self.maxs = X_u.max(axis = 0) # get max value of each feature
            X_l = normalize(X_l, [self.loLim, self.hiLim])
            X_u = normalize(X_u, [self.loLim, self.hiLim])
        else:
            self.isNorm = False
            self.mins = []
            self.maxs = []
        
        listLines = list()
        listInputSamplePoints = list();
        
        if self.isDraw:
            fig = plt.figure(0)
            plt.ion()
            if xX == 2:
                drawing_canvas = fig.add_subplot(1, 1, 1)
                drawing_canvas.axis([0, 1, 0, 1])
            else:
                drawing_canvas = Axes3D(fig)
                drawing_canvas.set_xlim3d(0, 1)
                drawing_canvas.set_ylim3d(0, 1)
                drawing_canvas.set_zlim3d(0, 1)
            
        while self.misclass > 0 and teta >= self.tMin:
            # for each input sample
            for i in range(yX):
                classOfX = patClassId[i]
                # draw input samples
                if self.isDraw:
                    if i == 0 and len(listInputSamplePoints) > 0:
                        # reset input point drawing
                        for point in listInputSamplePoints:
                            point.remove()
                    
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
                            inputPoint = drawing_canvas.plot(X_l[i, 0], X_l[i, 1], X_l[i, 2], color = color_, marker=marker_)
                        
                        #listInputSamplePoints.append(inputPoint)
                    else:
                        inputPoint = drawbox(np.asmatrix(X_l[i, 0:np.minimum(xX, 3)]), np.asmatrix(X_u[i, 0:np.minimum(xX, 3)]), drawing_canvas, color_)
                        
                    listInputSamplePoints.append(inputPoint[0])
                    plt.pause(self.delayConstant)
                    
                if self.V.size == 0:   # no model provided - starting from scratch
                    self.V = np.array([X_l[0]])
                    self.W = np.array([X_u[0]])
                    self.classId = np.array([patClassId[0]])
                    
                    if self.isDraw == True:
                        # draw hyperbox
                        box_color = 'k'
                        if patClassId[0] < len(mark_col):
                            box_color = mark_col[patClassId[0]]
                        
                        hyperbox = drawbox(np.asmatrix(self.V[0, 0:np.minimum(xX,3)]), np.asmatrix(self.W[0, 0:np.minimum(xX,3)]), drawing_canvas, box_color)
                        listLines.append(hyperbox[0])
                        plt.pause(self.delayConstant)

                else:
                    b = memberG(X_l[i], X_u[i], self.V, self.W, self.gamma)
                        
                    index = np.argsort(b);
                    bSort = b[index];
                    
                    #if bSort[-1] == 1 and (patClassId[i] == self.classId[index[-1]] or patClassId[i] == 0):
                        #classOfX = patClassId[i]
                    #else:
                    if bSort[-1] != 1 or (classOfX != self.classId[index[-1]] and classOfX != 0):
                        reversed_index = index[::-1]
                        adjust = False
                        for j in reversed_index:
                            # test violation of max hyperbox size and class labels
                            if ((np.maximum(self.W[j], X_u[i]) - np.minimum(self.V[j], X_l[i])) <= self.teta).min() == True and (classOfX == self.classId[j] or self.classId[j] == 0 or classOfX == 0):
                                # adjust the j-th hyperbox
                                self.V[j] = np.minimum(self.V[j], X_l[i])
                                self.W[j] = np.maximum(self.W[j], X_u[i])
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
                                    plt.pause(self.delayConstant)
                                    
                                break
                                
                        # if i-th sample did not fit into any existing box, create a new one
                        if not adjust:
                            self.V = np.vstack((self.V, X_l[i]))
                            self.W = np.vstack((self.W, X_u[i]))
                            self.classId = np.append(self.classId, classOfX)

                            if self.isDraw:
                                # handle drawing graph
                                box_color = 'k'
                                if self.classId[-1] < len(mark_col):
                                    box_color = mark_col[self.classId[-1]]
                                    
                                hyperbox = drawbox(np.asmatrix(X_l[i, 0:np.minimum(xX, 3)]), np.asmatrix(X_u[i, 0:np.minimum(xX, 3)]), drawing_canvas, box_color)
                                listLines.append(hyperbox[0])
                                plt.pause(self.delayConstant)
                                
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
                                            plt.pause(self.delayConstant)
                            
           						
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
#            plt.pause(self.delayConstant)
#                
        if self.isDraw:
            plt.show()

        return self
    
    def predict(self, Xl_Test, Xu_Test, patClassIdTest):
        """
        Perform classification
        
            result = predict(Xl_Test, Xu_Test, patClassIdTest)
        
        INPUT:
            Xl_Test             Test data lower bounds (rows = objects, columns = features)
            Xu_Test             Test data upper bounds (rows = objects, columns = features)
            patClassIdTest	     Test data class labels (crisp)
            
        OUTPUT:
            result        A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + sumamb           Number of objects with maximum membership in more than one class
                          + out              Soft class memberships
                          + mem              Hyperbox memberships
        """
        # Normalize testing dataset if training datasets were normalized
        if len(self.mins) > 0:
            noSamples = Xl_Test.shape[0]
            Xl_Test = self.loLim + (self.hiLim - self.loLim) * (Xl_Test - np.ones((noSamples, 1)) * self.mins) / (np.ones((noSamples, 1)) * (self.maxs - self.mins))
            Xu_Test = self.loLim + (self.hiLim - self.loLim) * (Xu_Test - np.ones((noSamples, 1)) * self.mins) / (np.ones((noSamples, 1)) * (self.maxs - self.mins))
            
            if Xl_Test.min() < self.loLim or Xu_Test.min() < self.loLim or Xl_Test.max() > self.hiLim or Xu_Test.max() > self.hiLim:
                print('Test sample falls ousitde', self.loLim, '-', self.hiLim, 'interval')
                return
            
        # do classification
        result = predict(self.V, self.W, self.classId, Xl_Test, Xu_Test, patClassIdTest, self.gamma, self.oper)
        
        return result
 
def string_to_boolean(st):
    if st == "True" or st == "true":
        return True
    elif st == "False" or st == "false":
        return False
    else:
        raise ValueError
        
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
    
    print('isDraw = ', isDraw, ' teta = ', teta, ' teta_min = ', teta_min, ' gamma = ', gamma, ' oper = ', oper, ' isNorm = ', isNorm, ' norm_range = ', norm_range)
    
    if sys.argv[1] == '1':
        training_file = sys.argv[2]
        testing_file = sys.argv[3]

        # Read training file
        Xtr, X_tmp, patClassIdTr, pat_tmp = loadDataset(training_file, 1, False)
        # Read testing file
        X_tmp, Xtest, pat_tmp, patClassIdTest = loadDataset(testing_file, 0, False)
    
        classifier = OnlineGFMM(gamma, teta, teta_min, isDraw, oper, isNorm, norm_range)
        classifier.fit(Xtr, Xtr, patClassIdTr)
    
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
   
        