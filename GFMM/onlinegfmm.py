# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 22:39:47 2018

@author: Thanh Tung Khuat

onlnGFMM - Online GFMM classifier (training core)

     OnlineGFMM(gamma, teta, tMin, isDraw, oper, V, W, classId)
  
   INPUT
     V				Hyperbox lower bounds for the model to be updated using new data
	  W				Hyperbox upper bounds for the model to be updated using new data
     classId		Hyperbox class labels (crisp)  for the model to be updated using new data
     gamma		Membership function slope (default: 1), datatype: array or scalar
     teta			Maximum hyperbox size (default: 1)
     tMin			Minimum value of Teta
     isDraw	   Progress plot flag (default: False)
     oper			Membership calculation operation: 'min' or 'prod' (default: 'min')

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from membershipcalc import memberG
from hyperboxadjustment import hyperboxOverlapTest, hyperboxContraction
from classification import predict
from drawinghelper import drawbox
from prepocessinghelper import loadDataset

class OnlineGFMM(object):
    
    def __init__(self, gamma = 1, teta = 1, tMin = 1, isDraw = False, oper = 'min', V = np.array([], dtype=np.float64), W = np.array([], dtype=np.float64), classId = np.array([], dtype=np.int16)):
        self.gamma = gamma
        self.teta = teta
        self.tMin = tMin
        self.V = V
        self.W = W
        self.classId = classId
        self.isDraw = isDraw
        self.oper = oper
        self.misclass = 1
        
    def fit(self, X_l, X_u, patClassId):
        """
        Training the classifier
        
         Xl			  Input data lower bounds (rows = objects, columns = features)
         Xu			  Input data upper bounds (rows = objects, columns = features)
         patClassId	  Input data class labels (crisp). patClassId[i] = 0 corresponds to an unlabeled item
        
        """
        print('--Online Learning--')
        yX, xX = X_l.shape
        teta = self.teta
        
        mark = np.array(['*', 'o', 'x', '+', '.', ',', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H', 'X', 'D', '|', '_'])
        mark_col = np.array(['r', 'g', 'b', 'y', 'c', 'm', 'k'])
        
        han = np.empty(yX, dtype=object)
        if self.isDraw:
            fig = plt.figure(0)
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
                    if i == 0:
                        # reset graph view
                        drawing_canvas.cla()
                    
                    color_ = 'k'
                    if classOfX < len(mark_col):
                        color_ = mark_col[classOfX]
                    
                    if (X_l[i, :] == X_u[i, :]).all():
                        marker_ = 'd'                   
                        if classOfX < len(mark):
                            marker_ = mark[classOfX]
                            
                        if xX == 2:
                            drawing_canvas.plot(X_l[i, 0], X_l[i, 1], color = color_, marker=marker_)
                        else:
                            drawing_canvas.plot(X_l[i, 0], X_l[i, 11], X_l[i, 2], color = color_, marker=marker_)
                    else:
                            drawbox(np.asmatrix(X_l[i, 0:np.minimum(xX, 3)]), np.asmatrix(X_u[i, 0:np.minimum(xX, 3)]), drawing_canvas, color_)
                        
                    plt.pause(0.001)
                    
                if self.V.size == 0:   # no model provided - starting from scratch
                    self.V = np.array([X_l[0]])
                    self.W = np.array([X_u[0]])
                    self.classId = np.array([patClassId[0]])
                    
                    if self.isDraw == True:
                        # draw hyperbox
                        box_color = 'k'
                        if patClassId[0] < len(mark_col):
                            box_color = mark_col[patClassId[0]]
                        
                        han[0] = drawbox(np.asmatrix(self.V[0, 0:np.minimum(xX,3)]), np.asmatrix(self.W[0, 0:np.minimum(xX,3)]), drawing_canvas, box_color)
                        plt.pause(0.001)

                else:
                    #print('Cheer!!!')
                    b = memberG(X_l[i], X_u[i], self.V, self.W, self.gamma)
                        
                    #print(b.shape)
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
                                    #drawing_canvas.lines.remove(han[j])
                                    box_color = 'k'
                                    if self.classId[j] < len(mark_col):
                                        box_color = mark_col[self.classId[j]]
                                        
                                    han[j] = drawbox(np.asmatrix(self.V[j, 0:np.minimum(xX, 3)]), np.asmatrix(self.W[j, 0:np.minimum(xX, 3)]), drawing_canvas, box_color)
                                    plt.pause(0.001)
                                    
                                break
                                
                        # if i-th sample did not fit into any existing box, create a new one
                        if not adjust:
                            self.V = np.vstack((self.V, X_l[i]))
                            self.W = np.vstack((self.W, X_u[i]))
                            self.classId = np.append(self.classId, classOfX)
                            #indOfWinner = b.size + 1
                            #adjust = True
                            #print('Add new')
                            if self.isDraw:
                                # handle drawing graph
                                box_color = 'k'
                                if self.classId[-1] < len(mark_col):
                                    box_color = mark_col[self.classId[-1]]
                                    
                                han[b.size + 1] = drawbox(np.asmatrix(X_l[i, 0:np.minimum(xX, 3)]), np.asmatrix(X_u[i, 0:np.minimum(xX, 3)]), drawing_canvas, box_color)
                                plt.pause(0.001)
                                
                        elif self.V.shape[0] > 1:
                            #print('Expend')
                            for ii in range(self.V.shape[0]):
                                if ii != indOfWinner:
                                    caseDim = hyperboxOverlapTest(self.V, self.W, indOfWinner, ii)		# overlap test
                                    
                                    if caseDim.size > 0 and self.classId[ii] != self.classId[indOfWinner]:
                                        self.V, self.W = hyperboxContraction(self.V, self.W, caseDim, ii, indOfWinner)
                                        if self.isDraw:
                                            # Handle graph drawing
                                            #drawing_canvas.artists.remove(han[ii, indOfWinner])
                                            boxii_color = boxwin_color = 'k'
                                            if self.classId[ii] < len(mark_col):
                                                boxii_color = mark_col[self.classId[ii]]
                                            
                                            if self.classId[indOfWinner] < len(mark_col):
                                                boxwin_color = mark_col[self.classId[indOfWinner]]
                                                
                                            han[[ii, indOfWinner]] = drawbox(self.V[[ii, indOfWinner], 0:np.minimum(xX, 3)], self.W[[ii, indOfWinner], 0:np.minimum(xX, 3)], drawing_canvas, [boxii_color, boxwin_color])
                                            plt.pause(0.001)
                            
           						
            teta = teta * 0.9           
            result = predict(self.V, self.W, self.classId, X_l, X_u, patClassId, self.gamma, self.oper)
            
            if result.summis > 0 and self.isDraw == True:
                # Handle drawing graph
                drawing_canvas.cla()
                color_ = np.empty(len(self.classId), dtype = object)
                for c in range(len(self.classId)):
                    color_[c] = mark_col[self.classId[c]]
                    
                drawbox(self.V[:, 0:np.minimum(xX, 3)], self.W[:, 0:np.minimum(xX, 3)], drawing_canvas, color_)
                plt.pause(0.0001)
                
        if self.isDraw:
            plt.show()

        return self
    
    def predict(self, Xl_Test, Xu_Test, patClassIdTest):
        """
        Perform classification
        
            result = predict(Xl_Test, Xu_Test, patClassIdTest)
        
        INPUT:
            Xl_Test:            Test data lower bounds (rows = objects, columns = features)
            Xu_Test:            Test data upper bounds (rows = objects, columns = features)
            patClassIdTest	      Test data class labels (crisp)
            
        OUTPUT:
            result        A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + sumamb           Number of objects with maximum membership in more than one class
                          + out              Soft class memberships
                          + mem              Hyperbox memberships
        """
        result = predict(self.V, self.W, self.classId, Xl_Test, Xu_Test, patClassIdTest, self.gamma, self.oper)
        
        return result
        
        
if __name__ == '__main__':
    Xtr, X_tmp, patClassIdTr, pat_tmp = loadDataset('synthetic_train.dat', 1, False)
    classifier = OnlineGFMM(1, 0.6, 0.6, True)
    classifier.fit(Xtr, Xtr, patClassIdTr)
    
    # Testing
    X_tmp, Xtest, pat_tmp, patClassIdTest = loadDataset('synthetic_test.dat', 0, False)
    print("-- Testing --")
    result = classifier.predict(Xtest, Xtest, patClassIdTest)
    print("Number of wrong predicted samples = ", result.summis)
    numTestSample = Xtest.shape[0]
    print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")
                    