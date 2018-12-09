# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:42:53 2018

@author: Thanh Tung Khuat

Online GFMM classifier (training core) using Pytorch

     Torch_OnlineGFMM(gamma, teta, tMin, isDraw, oper, V, W, classId, isNorm, norm_range)
  
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

import sys, os
sys.path.insert(0, os.path.pardir) 
import ast
import torch
torch.cuda.FloatTensor(1)
import numpy as np
import time
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

from functionhelper import float_def, long_def, is_Have_GPU, GPU_Computing_Threshold
from functionhelper.torch_membership_calc import torch_memberG, gpu_memberG
from functionhelper.torch_hyperboxadjustment import torch_hyperboxOverlapTest, torch_hyperboxContraction
from GFMM.classification import torch_predict
from functionhelper.hyperboxadjustment import hyperboxOverlapTest
from functionhelper.drawinghelper import drawbox
from functionhelper.preprocessinghelper import loadDataset, string_to_boolean
from GFMM.torch_basegfmmclassifier import Torch_BaseGFMMClassifier

class Torch_OnlineGFMM(Torch_BaseGFMMClassifier):
    
    def __init__(self, gamma = 1, teta = 1, tMin = 1, isDraw = False, oper = 'min', isNorm = False, norm_range = [0, 1], V = torch.FloatTensor([]), W = torch.FloatTensor([]), classId = torch.LongTensor([])):
        Torch_BaseGFMMClassifier.__init__(self, gamma, teta, isDraw, oper, isNorm, norm_range)
        
        self.tMin = tMin
        self.V = V
        self.W = W
        self.classId = classId
        self.misclass = 1
        
        
    def fit(self, X_l, X_u, patClassId):
        """
        Training the classifier
        
         Xl             Input data lower bounds (rows = objects, columns = features)
         Xu             Input data upper bounds (rows = objects, columns = features)
         patClassId     Input data class labels (crisp). patClassId[i] = 0 corresponds to an unlabeled item
        
        """
        print('--Online Learning--')
        
        if self.isNorm == True:
            X_l, X_u = self.dataPreprocessing(X_l, X_u)
        
        if isinstance(X_l, torch.Tensor) == False:
            # X_l = torch.cuda.FloatTensor(X_l)
            # X_u = torch.cuda.FloatTensor(X_u)
            # patClassId = torch.cuda.LongTensor(patClassId)
            # print('Conver data')
            # t1 = time.clock()
            X_l = torch.from_numpy(X_l).float()
            X_u = torch.from_numpy(X_u).float()
            patClassId = torch.from_numpy(patClassId).long()
#            t2 = time.clock()
#            print('Finish Conver data: ', t2 - t1)
        
        time_start = time.perf_counter()
        
        yX, xX = X_l.size()
        teta = self.teta
        
        mark = np.array(['*', 'o', 'x', '+', '.', ',', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H', 'X', 'D', '|', '_'])
        mark_col = np.array(['r', 'g', 'b', 'y', 'c', 'm', 'k'])
        
        listLines = list()
        listInputSamplePoints = list()
        
        if self.isDraw:
            drawing_canvas = self.initializeCanvasGraph("GFMM - Online learning", xX)
            
            if self.V.size(0) > 0:
                # draw existed hyperboxes
                color_ = np.array(['k'] * len(self.classId), dtype = object)
                for c in range(len(self.classId)):
                    if self.classId[c] < len(mark_col):
                        color_[c] = mark_col[self.classId[c]]
                
                hyperboxes = drawbox(self.V[:, 0:np.minimum(xX,3)].numpy(), self.W[:, 0:np.minimum(xX,3)].numpy(), drawing_canvas, color_)
                listLines.extend(hyperboxes)
                self.delay()
        
        self.misclass = 1
        isUsingGPU = False
        
        while self.misclass > 0 and teta >= self.tMin:
            # for each input sample
            for i in range(yX):
                if len(self.V) > 0 and is_Have_GPU and isUsingGPU == False and self.V.size(0) * self.V.size(1) >= GPU_Computing_Threshold:
                    self.V = self.V.cuda()
                    self.W = self.W.cuda()
                    self.classId = self.classId.cuda()
                    isUsingGPU = True
#                print('Sample: ', i)
                # draw input samples
                if self.isDraw:
                    if i == 0 and len(listInputSamplePoints) > 0:
                        # reset input point drawing
                        for point in listInputSamplePoints:
                            point.remove()
                        listInputSamplePoints.clear()
                    
                    color_ = 'k'
                    if patClassId[i] < len(mark_col):
                        color_ = mark_col[patClassId[i]]
                    
                    if (X_l[i, :] == X_u[i, :]).all():
                        marker_ = 'd'                   
                        if patClassId[i] < len(mark):
                            marker_ = mark[patClassId[i]]
                            
                        if xX == 2:
                            inputPoint = drawing_canvas.plot(X_l[i, 0], X_l[i, 1], color = color_, marker=marker_)
                        else:
                            inputPoint = drawing_canvas.plot([X_l[i, 0]], [X_l[i, 1]], [X_l[i, 2]], color = color_, marker=marker_)
                        
                        #listInputSamplePoints.append(inputPoint)
                    else:
                        inputPoint = drawbox(np.asmatrix(X_l[i, 0:np.minimum(xX, 3)].numpy()), np.asmatrix(X_u[i, 0:np.minimum(xX, 3)].numpy()), drawing_canvas, color_)
                        
                    listInputSamplePoints.append(inputPoint[0])
                    self.delay()
                    
                if self.V.size(0) == 0:   # no model provided - starting from scratch
                    #print('Initial data')
                    #t1 = time.clock()
                    self.V = X_l[0].reshape(1, -1) # torch.DoubleTensor(X_l[0]).to(device)
                    self.W = X_u[0].reshape(1, -1) # torch.DoubleTensor(X_u[0]).to(device)
                    self.classId = torch.LongTensor([patClassId[0]]) # torch.DoubleTensor([patClassId[0]]).to(device)
                    #t2 = time.clock()
                    #print('Finish Initial data: ', t2 - t1)
                    
                    if self.isDraw == True:
                        # draw hyperbox
                        box_color = 'k'
                        if patClassId[0] < len(mark_col):
                            box_color = mark_col[patClassId[0]]
                        
                        hyperbox = drawbox(np.asmatrix(self.V[0, 0:np.minimum(xX,3)].numpy()), np.asmatrix(self.W[0, 0:np.minimum(xX,3)].numpy()), drawing_canvas, box_color)
                        listLines.append(hyperbox[0])
                        self.delay()

                else:
                    # print('V === ', self.V)
                    # print('W === ', self.W)
#                    print('Compute membership')
#                    t1 = time.clock()
                    if isUsingGPU == False:
                        Xl_cur = X_l[i]
                        Xu_cur = X_u[i]
                        classOfX = patClassId[i]
                        b = torch_memberG(Xl_cur, Xu_cur, self.V, self.W, self.gamma)
                    else:
                        Xl_cur = X_l[i].cuda()
                        Xu_cur = X_u[i].cuda()
                        classOfX = patClassId[i].cuda()
                        
                        b = gpu_memberG(Xl_cur, Xu_cur, self.V, self.W, self.gamma)
#                    t2 = time.clock()
#                    print('Finish computing membership: ', t2 - t1)
                    # print(b)
                    bSort, index = torch.sort(b, descending=True)
                    # print('index = ', index)
                    # print('Self-class: ', self.classId)
                    # print(' i===', i)
                    
                    if bSort[0] != 1 or (classOfX != self.classId[index[0]] and classOfX != 0):
                        adjust = False
                        for j in index:
                            # test violation of max hyperbox size and class labels
                            if (classOfX == self.classId[j] or self.classId[j] == 0 or classOfX == 0) and ((torch.max(self.W[j], Xu_cur).float() - torch.min(self.V[j], Xl_cur).float()) <= teta).all() == True:
                                # adjust the j-th hyperbox
                                self.V[j] = torch.min(self.V[j], Xl_cur)
                                self.W[j] = torch.max(self.W[j], Xu_cur)
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
                                    
                                    if isUsingGPU == False:
                                        hyperbox = drawbox(np.asmatrix(self.V[j, 0:np.minimum(xX, 3)].numpy()), np.asmatrix(self.W[j, 0:np.minimum(xX, 3)].numpy()), drawing_canvas, box_color)
                                    else:
                                        hyperbox = drawbox(np.asmatrix(self.V[j, 0:np.minimum(xX, 3)].cpu().numpy()), np.asmatrix(self.W[j, 0:np.minimum(xX, 3)].cpu().numpy()), drawing_canvas, box_color)
                                        
                                    listLines[j] = hyperbox[0]
                                    self.delay()
                                    
                                break
                               
                        # if i-th sample did not fit into any existing box, create a new one
                        if not adjust:
#                            print('Create new hyperbox')
#                            t1 = time.clock()
                            self.V = torch.cat((self.V, Xl_cur.reshape(1, -1)), 0)
                            self.W = torch.cat((self.W, Xu_cur.reshape(1, -1)), 0)
                            if isUsingGPU == False:
                                self.classId = torch.cat((self.classId, torch.LongTensor([classOfX])), 0)
                            else:
                                self.classId = torch.cat((self.classId, torch.cuda.LongTensor([classOfX])), 0)
#                            t2 = time.clock()
#                            print('Finish compute new hyperbox: ', t2 - t1)
                            
                            if self.isDraw:
                                # handle drawing graph
                                box_color = 'k'
                                if self.classId[-1] < len(mark_col):
                                    box_color = mark_col[self.classId[-1]]
                                
                                if isUsingGPU == False:
                                    hyperbox = drawbox(np.asmatrix(X_l[i, 0:np.minimum(xX, 3)].numpy()), np.asmatrix(X_u[i, 0:np.minimum(xX, 3)].numpy()), drawing_canvas, box_color)
                                else:
                                    hyperbox = drawbox(np.asmatrix(X_l[i, 0:np.minimum(xX, 3)].cpu().numpy()), np.asmatrix(X_u[i, 0:np.minimum(xX, 3)].cpu().numpy()), drawing_canvas, box_color)
                                    
                                listLines.append(hyperbox[0])
                                self.delay()
                                
                        elif self.V.size(0) > 1:
                            for ii in range(self.V.size(0)):
                                if ii != indOfWinner and self.classId[ii] != self.classId[indOfWinner]:
#                                    print('Overlap Test')
#                                    t1 = time.clock()
                                    caseDim = torch_hyperboxOverlapTest(self.V, self.W, indOfWinner, ii)		# overlap test
#                                    t2 = time.clock()
#                                    print('Finish overlap test: ', t2 - t1)
                                    
                                    if len(caseDim) > 0:
#                                        print('Hyperbox Contraction')
#                                        t1 = time.clock()
                                        self.V, self.W = torch_hyperboxContraction(self.V, self.W, caseDim, ii, indOfWinner)
#                                        t2 = time.clock()
#                                        print('Finish hyperbox Contaction: ', t2 - t1)
                                        
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
                                            
                                            if isUsingGPU == False:
                                                hyperboxes = drawbox(self.V[[ii, indOfWinner], 0:np.minimum(xX, 3)].numpy(), self.W[[ii, indOfWinner], 0:np.minimum(xX, 3)].numpy(), drawing_canvas, [boxii_color, boxwin_color])
                                            else:
                                                hyperboxes = drawbox(self.V[[ii, indOfWinner], 0:np.minimum(xX, 3)].cpu().numpy(), self.W[[ii, indOfWinner], 0:np.minimum(xX, 3)].cpu().numpy(), drawing_canvas, [boxii_color, boxwin_color])      
                                            
                                            listLines[ii] = hyperboxes[0]
                                            listLines[indOfWinner] = hyperboxes[1]                                      
                                            self.delay()
                            
           						
            teta = teta * 0.9
            if teta >= self.tMin:
                result = torch_predict(self.V, self.W, self.classId, X_l, X_u, patClassId, self.gamma, self.oper)
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
        
        time_end = time.perf_counter()
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
        
    classifier = Torch_OnlineGFMM(gamma, teta, teta_min, isDraw, oper, isNorm, norm_range)
    classifier.fit(Xtr, Xtr, patClassIdTr)
    
    # Testing
    print("-- Testing --")
    result = classifier.predict_torch(Xtest, Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")
        print("Training time = ", classifier.elapsed_training_time)
        

