# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 16:28:37 2018

@author: Thanh Tung Khuat

Implementation of information representation based multi-layer classifier using GFMM

Note: Currently, all samples in the dataset must be normalized to the range of [0, 1] before using this class

The validation process and testing process in this code are run in parallel
"""
import sys, os
sys.path.insert(0, os.path.pardir)

import numpy as np
import math
import ast
import time
import multiprocessing
from functionhelper.bunchdatatype import Bunch
from functionhelper.membershipcalc import memberG, asym_similarity_one_many
from functionhelper.preprocessinghelper import read_file_in_chunks_group_by_label, read_file_in_chunks, string_to_boolean, loadDataset
from functionhelper.hyperboxadjustment import isOverlap, hyperboxOverlapTest, modifiedIsOverlap, hyperboxContraction
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn import metrics

def get_num_cpu_cores():
    num_cores = multiprocessing.cpu_count()
    if num_cores >= 4:
        num_cores = num_cores - 2
    return num_cores

class Info_Presentation_Multi_Layer_Classifier_GFMM(object):
    
    def __init__(self, teta = [0.1, 0.5], gamma = 1, simil_thres = 0.5, oper = 'min'):
        self.gamma = gamma
        self.teta_onl = teta[0]
        self.higher_teta = teta[1:]
        
        self.oper = oper
        self.simil_thres = simil_thres
        
        
    def homogeneous_hyperbox_expansion(self, X_l, X_u, patClassId, current_hyperboxes):
        """
            Expand current hyperboxes to cover input patterns, all input samples have the same label with each other as well as current hyperboxes (if exists)
            Update the number of patterns contained in the hyperboxes and their centroids of samples
            
                INPUT
                    Xl                      Input data lower bounds (rows = objects, columns = features)
                    Xu                      Input data upper bounds (rows = objects, columns = features)
                    patClassId              Input data class labels (crisp). patClassId[i] = 0 corresponds to an unlabeled item
        
                    current_hyperboxes      A list of current hyperboxes in the Bunch datatype (properties: lower, upper, classId, no_pat, centroid)
                    
                OUTPUT
                    result                  A bunch data size with lower and upper bounds, class labels of hyperboxes
        """
        yX = X_l.shape[0]
        V = current_hyperboxes.lower
        W = current_hyperboxes.upper
        classId = current_hyperboxes.classId
        no_Pats = current_hyperboxes.no_pat
        centroid = current_hyperboxes.centroid
        # for each input sample
        for i in range(yX):
            classOfX = patClassId[i]
                
            if V.size == 0:   # no model provided - starting from scratch
                V = np.array([X_l[i]])
                W = np.array([X_u[i]])
                classId = np.array([patClassId[i]])
                no_Pats = np.array([1])
                centroid = np.array([(X_l[i] + X_u[i]) / 2])
            else:
                b = memberG(X_l[i], X_u[i], V, W, self.gamma, self.oper)
                    
                index = np.argsort(b)[::-1]
                bSort = b[index];
                
                if bSort[0] != 1:
                    adjust = False
                    for j in index:
                        # test violation of max hyperbox size and class labels
                        if ((np.maximum(W[j], X_u[i]) - np.minimum(V[j], X_l[i])) <= self.teta_onl).all() == True:
                            # adjust the j-th hyperbox
                            V[j] = np.minimum(V[j], X_l[i])
                            W[j] = np.maximum(W[j], X_u[i])
                            
                            no_Pats[j] = no_Pats[j] + 1
                            centroid[j] = centroid[j] + (((X_l[i] + X_u[i]) / 2) - centroid[j]) / no_Pats[j]                            
                            
                            adjust = True
                            if classOfX != 0 and classId[j] == 0:
                                classId[j] = classOfX               
                                
                            break
                           
                    # if i-th sample did not fit into any existing box, create a new one
                    if not adjust:
                        V = np.concatenate((V, X_l[i].reshape(1, -1)), axis = 0)
                        W = np.concatenate((W, X_u[i].reshape(1, -1)), axis = 0)
                        classId = np.concatenate((classId, [classOfX]))
                        no_Pats = np.concatenate((no_Pats, [1]))
                        new_Central_Sample = (X_l[i] + X_u[i]) / 2
                        centroid = np.concatenate((centroid, new_Central_Sample.reshape(1, -1)), axis = 0)
                    
        return Bunch(lower=V, upper=W, classId=classId, no_pat=no_Pats, centroid=centroid)
    
    
    def heterogeneous_hyperbox_expansion(self, X_l, X_u, patClassId, current_hyperboxes):
        """
            Expand current hyperboxes to cover input patterns, input samples contains different labels
            Update the number of patterns contained in the hyperboxes and their centroids of samples
            
                INPUT
                    Xl                      Input data lower bounds (rows = objects, columns = features)
                    Xu                      Input data upper bounds (rows = objects, columns = features)
                    patClassId              Input data class labels (crisp). patClassId[i] = 0 corresponds to an unlabeled item
        
                    current_hyperboxes      A list of current hyperboxes in the Bunch datatype (properties: lower, upper, classId, no_pat, centroid)
                    
                OUTPUT
                    result                  A bunch data size with lower and upper bounds, class labels of hyperboxes
        """
        yX = X_l.shape[0]
        V = current_hyperboxes.lower
        W = current_hyperboxes.upper
        classId = current_hyperboxes.classId
        no_Pats = current_hyperboxes.no_pat
        centroid = current_hyperboxes.centroid
        # for each input sample
        for i in range(yX):
            classOfX = patClassId[i]

            if V.size == 0:   # no model provided - starting from scratch
                V = np.array([X_l[0]])
                W = np.array([X_u[0]])
                classId = np.array([patClassId[0]])
                no_Pats = np.array([1])
                centroid = np.array([(X_l[0] + X_u[0]) / 2])

            else:
                id_lb_sameX = np.logical_or(classId == classOfX, classId == 0)
                
                if id_lb_sameX.any() == True: 
                    V_sameX = V[id_lb_sameX]
                    W_sameX = W[id_lb_sameX]
                    lb_sameX = classId[id_lb_sameX]
                    id_range = np.arange(len(classId))
                    id_processing = id_range[id_lb_sameX]

                    b = memberG(X_l[i], X_u[i], V_sameX, W_sameX, self.gamma, self.oper)
                    index = np.argsort(b)[::-1]
                    bSort = b[index]
                
                    if bSort[0] != 1 or (classOfX != lb_sameX[index[0]] and classOfX != 0):
                        adjust = False
                        for j in id_processing[index]:
                            # test violation of max hyperbox size and class labels
                            if (classOfX == classId[j] or classId[j] == 0 or classOfX == 0) and ((np.maximum(W[j], X_u[i]) - np.minimum(V[j], X_l[i])) <= self.teta_onl).all() == True:
                                # adjust the j-th hyperbox
                                V[j] = np.minimum(V[j], X_l[i])
                                W[j] = np.maximum(W[j], X_u[i])
                                no_Pats[j] = no_Pats[j] + 1
                                centroid[j] = centroid[j] + (((X_l[i] + X_u[i]) / 2) - centroid[j]) / no_Pats[j]                               
                            
                                adjust = True
                                if classOfX != 0 and classId[j] == 0:
                                    classId[j] = classOfX

                                break

                        # if i-th sample did not fit into any existing box, create a new one
                        if not adjust:
                            V = np.concatenate((V, X_l[i].reshape(1, -1)), axis = 0)
                            W = np.concatenate((W, X_u[i].reshape(1, -1)), axis = 0)
                            classId = np.concatenate((classId, [classOfX]))
                            no_Pats = np.concatenate((no_Pats, [1]))
                            new_Central_Sample = (X_l[i] + X_u[i]) / 2
                            centroid = np.concatenate((centroid, new_Central_Sample.reshape(1, -1)), axis = 0)

                else:
                    # new class lable => create new pattern
                    V = np.concatenate((V, X_l[i].reshape(1, -1)), axis = 0)
                    W = np.concatenate((W, X_u[i].reshape(1, -1)), axis = 0)
                    classId = np.concatenate((classId, [classOfX]))
                    no_Pats = np.concatenate((no_Pats, [1]))
                    new_Central_Sample = (X_l[i] + X_u[i]) / 2
                    centroid = np.concatenate((centroid, new_Central_Sample.reshape(1, -1)), axis = 0)
   
        return Bunch(lower=V, upper=W, classId=classId, no_pat=no_Pats, centroid=centroid)
    
    
    def homogeneous_worker_distribution_chunk_by_class(self, chunk_data, dic_current_hyperboxes, nprocs):
        """
            Distribute data in the current chunk to each worker according to class labels in turn
            
                INPUT
                    chunk_data              a dictionary contains input data with key being label and value being respective bunch data (properties: data, label)
                    dic_current_hyperboxes  a dictionary contains current coordinates of hyperboxes with labels as keys and values being a list of nprocs bunches of hyperboxes 
                    nprocs                  number of processes needs to be generated
        
                OUTPUT
                    dic_results             a dictionary contains new coordinates of hyperboxes with labels as keys and values being a list of nprocs bunches of hyperboxe
        """
        dic_results = {}
        with ProcessPoolExecutor(max_workers=nprocs) as executor:
            for key in chunk_data:
                futures = []
                # get list of current hyperboxes or initialize empty list if not exist list or input key
                if len(dic_current_hyperboxes) > 0 and (key in dic_current_hyperboxes):
                    boxes = dic_current_hyperboxes[key]
                else:
                    boxes = np.empty(nprocs, dtype=Bunch)
                    for j in range(nprocs):
                        boxes[j] = Bunch(lower=np.array([]), upper=np.array([]), classId=np.array([]), no_pat=0, centroid=np.array([]))
                    
                values = chunk_data[key]
                num_samples = len(values.data)
                if num_samples >= nprocs:
                    chunksize = int(math.ceil(num_samples / float(nprocs)))
                    
                    for i in range(nprocs):
                        X_l = values.data[(chunksize * i) : (chunksize * (i + 1))]
                        X_u = values.data[(chunksize * i) : (chunksize * (i + 1))]
                        patClassId = values.label[(chunksize * i) : (chunksize * (i + 1))]
                
                        futures.append(executor.submit(self.homogeneous_hyperbox_expansion, X_l, X_u, patClassId, boxes[i]))
                        
                else:
                    futures.append(executor.submit(self.homogeneous_hyperbox_expansion, values, boxes[0]))
                    
                # Instruct workers to process results as they come, when all are completed
                as_completed(futures) # wait all workers completed
                lst_current_boxes = []
                for future in futures:
                    lst_current_boxes.append(future.result())
                    
                dic_results[key] = lst_current_boxes
        
        return dic_results
    
    def heterogeneous_worker_distribution_chunk(self, lst_chunk_data, lst_current_hyperboxes, nprocs):
        """
            Distribute data in the current chunk to each worker according to the order of patterns
            
                INPUT
                    lst_chunk_data          a list contains input data with key being label and value being respective bunch data (properties: data, label)
                    lst_current_hyperboxes  a list contains current coordinates of hyperboxes (the number of hyperboxes is respective to the number of init cores)
                    nprocs                  number of processes needs to be generated
        
                OUTPUT
                    lst_result              a list of newly generated coordinates of hyperboxes
        """
        lst_results = []
        futures = []
        if len(lst_current_hyperboxes) == 0:
            lst_current_hyperboxes = np.empty(nprocs, dtype=Bunch)
            for j in range(nprocs):
                lst_current_hyperboxes[j] = Bunch(lower=np.array([]), upper=np.array([]), classId=np.array([]), no_pat=0, centroid=np.array([]))
                    
        with ProcessPoolExecutor(max_workers=nprocs) as executor:
            chunksize = int(math.ceil(len(lst_chunk_data.label) / float(nprocs)))
            for i in range(nprocs):
                X_l = lst_chunk_data.data[(chunksize * i) : (chunksize * (i + 1))]
                X_u = lst_chunk_data.data[(chunksize * i) : (chunksize * (i + 1))]
                patClassId = lst_chunk_data.label[(chunksize * i) : (chunksize * (i + 1))]
               
                futures.append(executor.submit(self.heterogeneous_hyperbox_expansion, X_l, X_u, patClassId, lst_current_hyperboxes[i]))
                        
            # Instruct workers to process results as they come, when all are completed
            as_completed(futures) # wait all workers completed:
            for future in futures:
                lst_results.append(future.result())
                    
        return lst_results
    
    
    def removeContainedHyperboxes_UpdateCentroid(self):
        """
            Remove all hyperboxes contained in other hyperboxes with the same class label and update centroids of larger hyperboxes
            This operation is performed on the values of lower and upper bounds, labels, and instance variables
        """
        numBoxes = len(self.classId)
        indtokeep = np.ones(numBoxes, dtype=np.bool) # position of all hyperboxes kept
        no_removed_boxes = 0
        
        for i in range(numBoxes):
            # Filter hypeboxes with the sample label as hyperbox i
            id_hyperbox_same_label = self.classId == self.classId[i]
            id_hyperbox_same_label[i] = False # remove hyperbox i
            if id_hyperbox_same_label.any() == True:
                # exist at least one hyperbox with the same label as hyperbox i
                V_same = self.V[id_hyperbox_same_label]
                W_same = self.W[id_hyperbox_same_label]
                
                memValue = memberG(self.V[i], self.W[i], V_same, W_same, self.gamma, self.oper)
                equal_one_index = memValue == 1
                
                if np.sum(equal_one_index) > 0:
                    original_index = np.arange(0, numBoxes)
                    original_index_same_label = original_index[id_hyperbox_same_label]
                    
                    index_Parent_Hyperbox = original_index_same_label[np.nonzero(equal_one_index)[0]] # Find indices of hyperboxes that contain hyperbox i
                    
                    isIncluded = len(index_Parent_Hyperbox) > 0
                    
                    if isIncluded == True:
                        indtokeep[i] = False
                        no_removed_boxes = no_removed_boxes + 1
                        # Update centroid of larger hyperbox
                        
                        if len(index_Parent_Hyperbox) == 1:
                            parent_selection = index_Parent_Hyperbox[0]
                            
                        elif len(index_Parent_Hyperbox) > 1:
                            # Compute the distance from the centroid of hyperbox i to centroids of other hyperboxes and choose the hyperbox with the smallest distance to merge
                            min_dis = np.linalg.norm(self.centroid[i] - self.centroid[index_Parent_Hyperbox[0]])
                            parent_selection = index_Parent_Hyperbox[0]
                            for jj in range(1, len(index_Parent_Hyperbox)):
                                dist = np.linalg.norm(self.centroid[i] - self.centroid[index_Parent_Hyperbox[jj]])
                                if min_dis < dist:
                                    min_dis = dist
                                    parent_selection = index_Parent_Hyperbox[jj]
                                    
                        # Merge centroids and number of hyperboxes                       
                        self.centroid[parent_selection] = (self.no_pat[parent_selection] * self.centroid[parent_selection] + self.no_pat[i] * self.centroid[i]) / (self.no_pat[i] + self.no_pat[parent_selection])
                        self.no_pat[parent_selection] = self.no_pat[parent_selection] + self.no_pat[i]
        
        # remove hyperboxes contained in other hyperboxes
        self.V = self.V[indtokeep, :]
        self.W = self.W[indtokeep, :]
        self.classId = self.classId[indtokeep]
        self.centroid = self.centroid[indtokeep]
        self.no_pat = self.no_pat[indtokeep]
        self.no_contained_boxes = no_removed_boxes
    
    def predict_val(self, XlT, XuT, patClassIdTest, no_predicted_samples_hyperboxes):
        """
        GFMM classification for validation (validation routine) with hyperboxes stored in self. V, W, classId, centroid, no_pat
    
          result = predict_val(XlT,XuT,patClassIdTest)
    
            INPUT
              XlT               Test data lower bounds (rows = objects, columns = features)
              XuT               Test data upper bounds (rows = objects, columns = features)
              patClassIdTest    Test data class labels (crisp)
              no_predicted_samples_hyperboxes         A matrix contains the number of right and wrong predicted samples of current hyperboxes, column 1: right, column 2: wrong
              
           OUTPUT
              A matrix contains number of samples predicted right and wrong in hyperboxes (first column: right, second column: wrong)
    
        """
    
        #initialization
        yX = XlT.shape[0]
        mem = np.zeros((yX, self.V.shape[0]))
        # classifications
        for i in range(yX):
            mem[i, :] = memberG(XlT[i, :], XuT[i, :], self.V, self.W, self.gamma, self.oper) # calculate memberships for all hyperboxes
            bmax = mem[i,:].max()	                                          # get max membership value
            maxVind = np.nonzero(mem[i,:] == bmax)[0]                         # get indexes of all hyperboxes with max membership
            
            if len(maxVind) == 1:
                # Only one hyperbox with the highest membership function
                
                if self.classId[maxVind[0]] == patClassIdTest[i]:
                    no_predicted_samples_hyperboxes[maxVind[0], 0] = no_predicted_samples_hyperboxes[maxVind[0], 0] + 1                 
                else:
                    no_predicted_samples_hyperboxes[maxVind[0], 1] = no_predicted_samples_hyperboxes[maxVind[0], 1] + 1
            else:
                # More than one hyperbox with highest membership => compare with centroid
                centroid_input_pat = (XlT[i] + XuT[i]) / 2
                id_min = maxVind[0]
                min_dist = np.linalg.norm(self.centroid[id_min] - centroid_input_pat)

                for j in range(1, len(maxVind)):
                    id_j = maxVind[j]
                    dist_j = np.linalg.norm(self.centroid[id_j] - centroid_input_pat)
                    
                    if dist_j < min_dist or (dist_j == min_dist and self.no_pat[id_j] > self.no_pat[id_min]):
                        id_min = id_j
                        min_dist = dist_j
                        
                if self.classId[id_min] != patClassIdTest[i] and patClassIdTest[i] != 0:
                    no_predicted_samples_hyperboxes[id_min, 1] = no_predicted_samples_hyperboxes[id_min, 1] + 1
                else:
                    no_predicted_samples_hyperboxes[id_min, 0] = no_predicted_samples_hyperboxes[id_min, 0] + 1
                    
        return no_predicted_samples_hyperboxes
    
    def pruningHandling(self, valFile_Path, nprocs = 4, accuracy_threshold = 0.5):
        """
            Pruning for hyperboxes in the current lists: V, W, classid, centroid
            Criteria:   The accuracy rate < 0.5
                        
                INPUT
                    nprocs              The number of processes
                    valFile_Path        The path to the validation file including filename and its extension
                    accuracy_threshold  The minimum accuracy for each hyperbox
                    
        """
        
        # pruning using validation set
        currenNoHyperbox = len(self.classId)
        if currenNoHyperbox > 0:
            # index_Kept = np.ones(currenNoHyperbox).astype(bool) # recompute the marking matrix
            # init two lists containing number of patterns classified correctly and incorrectly for each hyperbox
            no_predicted_samples_hyperboxes = np.zeros((len(self.classId), 2))
            
            lst_results = []
            futures = []
            
            _, Xval, _, patClassIdVal = loadDataset(valFile_Path, 0, False)
            
            with ProcessPoolExecutor(max_workers=nprocs) as executor:
                chunksize = int(math.ceil(len(patClassIdVal) / float(nprocs)))
                for i in range(nprocs):
                    XlT = Xval[(chunksize * i) : (chunksize * (i + 1))]
                    XuT = Xval[(chunksize * i) : (chunksize * (i + 1))]
                    patClassId = patClassIdVal[(chunksize * i) : (chunksize * (i + 1))]
                   
                    futures.append(executor.submit(self.predict_val, XlT, XuT, patClassId, no_predicted_samples_hyperboxes))
                # Instruct workers to process results as they come, when all are completed
                as_completed(futures) # wait all workers completed:
                for future in futures:
                    lst_results.append(future.result())
            
            no_predicted_samples_hyperboxes = np.zeros((len(self.classId), 2))
            # merging results returned from different processes
            for pr in range(len(lst_results)):
                no_predicted_samples_hyperboxes = no_predicted_samples_hyperboxes + lst_results[pr]
                
            # pruning handling based on the validation results
            tmp_no_box = no_predicted_samples_hyperboxes.shape[0]
            accuracy_larger_half = np.zeros(tmp_no_box).astype(np.bool)
            for i in range(tmp_no_box):
                if (no_predicted_samples_hyperboxes[i, 0] + no_predicted_samples_hyperboxes[i, 1] != 0) and no_predicted_samples_hyperboxes[i, 0] / (no_predicted_samples_hyperboxes[i, 0] + no_predicted_samples_hyperboxes[i, 1]) >= accuracy_threshold:
                    accuracy_larger_half[i] = True
            # Pruning
            self.V = self.V[accuracy_larger_half]
            self.W = self.W[accuracy_larger_half]
            self.classId = self.classId[accuracy_larger_half]
            self.centroid = self.centroid[accuracy_larger_half]
            self.no_pat = self.no_pat[accuracy_larger_half]
        
    
    def granular_phase_one_classifier(self, dataFilePath, chunk_size, type_chunk = 1, isPruning = False, valFile_Path = '', accuracyPerBox = 0.5, XlT = None, XuT = None, patClassIdTest = None, file_object_save = None):
        """
            This method is to read the dataset in chunks and build base hyperboxes from the input data
            
                INPUT
                    dataFilePath        The path to the training dataset file including file name and its extension
                    chunk_size          The size of each reading chunk to be handled
                    type_chunk          The type of data contained in each chunk:
                                            + 1:            heterogeneous data with different class label
                                            + otherwise:    data are grouped by class labels
                    isPruning           True: apply the pruning process
                                        False: not use the pruning process
                    valFile_Path        The path to the validation file including filename and its extension
                    accuracyPerBox      Minimum accuracy of each hyperbox w.r.t validation set
                    XlT               Test data lower bounds (rows = objects, columns = features)
                    XuT               Test data upper bounds (rows = objects, columns = features)
                    patClassIdTest    Test data class labels (crisp)
                    
        """
        chunk_id = 0
        nprocs = get_num_cpu_cores() # get number of cores in cpu for handling data
        print("No. cores =", nprocs)
        if file_object_save != None:
            file_object_save.write("No. cores = %d \n" % nprocs)
        # Initialize hyperboxes for each core
        if type_chunk == 1:
            current_hyperboxes = [] # empty list
        else:
            current_hyperboxes = {} # empty hashtable
              
        time_start = time.perf_counter()
        while True:
            chunk_data = read_file_in_chunks(dataFilePath, chunk_id, chunk_size) if type_chunk == 1 else read_file_in_chunks_group_by_label(dataFilePath, chunk_id, chunk_size)
            if chunk_data != None:
                if type_chunk == 1:
                    current_hyperboxes = self.heterogeneous_worker_distribution_chunk(chunk_data, current_hyperboxes, nprocs)
                else:
                    current_hyperboxes = self.homogeneous_worker_distribution_chunk_by_class(chunk_data, current_hyperboxes, nprocs)
                
                chunk_id = chunk_id + 1
            else:
                break
        
        # Merge all generated hyperboxes and then remove all hyperboxes insider larger hyperboxes and update their centroids
        if type_chunk == 1:
            self.V = current_hyperboxes[0].lower
            self.W = current_hyperboxes[0].upper
            self.classId = current_hyperboxes[0].classId
            self.no_pat = current_hyperboxes[0].no_pat
            self.centroid = current_hyperboxes[0].centroid
            
            num_Eles = len(current_hyperboxes)
            for kk in range(1, num_Eles):
                self.V = np.concatenate((self.V, current_hyperboxes[kk].lower), axis=0)
                self.W = np.concatenate((self.W, current_hyperboxes[kk].upper), axis=0)
                self.classId = np.concatenate((self.classId, current_hyperboxes[kk].classId))
                self.no_pat = np.concatenate((self.no_pat, current_hyperboxes[kk].no_pat))
                self.centroid = np.concatenate((self.centroid, current_hyperboxes[kk].centroid), axis=0)
        
        else:
            self.V = []
            for key in current_hyperboxes:
                for value in current_hyperboxes[key]:
                    if len(self.V) == 0:
                        self.V = value.lower
                        self.W = value.upper
                        self.classId = value.classId
                        self.no_pat = value.no_pat
                        self.centroid = value.centroid
                    else:
                        if len(value.lower) > 0:
                            self.V = np.concatenate((self.V, value.lower), axis=0)
                            self.W = np.concatenate((self.W, value.upper), axis=0)
                            self.classId = np.concatenate((self.classId, value.classId))
                            self.no_pat = np.concatenate((self.no_pat, value.no_pat))
                            self.centroid = np.concatenate((self.centroid, value.centroid), axis=0)
        
        # delete hyperboxes contained in other hyperboxes and update the centroids of larger hyperboxes
        self.removeContainedHyperboxes_UpdateCentroid()
        numBoxes_before_pruning = len(self.classId)
        self.phase1_elapsed_training_time = time.perf_counter() - time_start
        self.training_time_before_pruning = self.phase1_elapsed_training_time
        
        if (XlT is not None) and (len(self.classId) > 0):
            numTestSample = XlT.shape[0]
            
            result_testing = self.parallel_predict_test(nprocs, XlT, XuT, patClassIdTest)
            if (result_testing is not None) and file_object_save is not None:
                file_object_save.write("Phase 1 before pruning: \n")
                file_object_save.write("Number of testing samples = %d \n" % numTestSample)
                file_object_save.write("Number of wrong predicted samples = %d \n" % result_testing.summis)
                file_object_save.write("Error Rate = %f \n" % (np.round(result_testing.summis / numTestSample * 100, 4)))
                file_object_save.write("AUC = %f \n" % result_testing.auc)
                file_object_save.write("No. samples use centroid for prediction = %d \n" % result_testing.use_centroid)
                file_object_save.write("No. samples use centroid but wrong prediction = %d \n" % result_testing.use_centroid_wrong)
            
        if isPruning:
            time_start = time.perf_counter()
            self.pruningHandling(valFile_Path, nprocs, accuracyPerBox)
            numBoxes_after_pruning = len(self.classId)
            self.phase1_elapsed_training_time = self.phase1_elapsed_training_time + (time.perf_counter() - time_start)
            
            if (XlT is not None) and len(self.classId) > 0:
                result_testing = self.parallel_predict_test(nprocs, XlT, XuT, patClassIdTest)
                if result_testing is not None and file_object_save is not None:
                    file_object_save.write("Phase 1 after pruning: \n")
                    file_object_save.write("Number of wrong predicted samples = %d \n" % result_testing.summis)
                    file_object_save.write("Error Rate = %f \n" % (np.round(result_testing.summis / numTestSample * 100, 4)))
                    file_object_save.write("AUC = %f \n" % result_testing.auc)
                    file_object_save.write("No. samples use centroid for prediction = %d \n" % result_testing.use_centroid)
                    file_object_save.write("No. samples use centroid but wrong prediction = %d \n" % result_testing.use_centroid_wrong)              
        
        
        if file_object_save is not None:
            file_object_save.write("No. hyperboxes before pruning: %d \n" % numBoxes_before_pruning)
        if isPruning and file_object_save is not None:
            file_object_save.write("No. hyperboxes after pruning: %d \n" % numBoxes_after_pruning)
        
        if file_object_save is not None:
            file_object_save.write('Phase 1 running time = %f \n' % self.phase1_elapsed_training_time)
            file_object_save.write('Running time before pruning = %f \n' % self.training_time_before_pruning)
        
        return self
    
    
#    def granular_phase_two_classifier(self, isAllowedOverlap = False):
#        """
#            Phase 2 in the classifier: using agglomerative learning to aggregate smaller hyperboxes with the same class
#            
#                granular_phase_two_classifier(isAllowedOverlap)
#                
#                INPUT
#                    isAllowedOverlap        + True: the aggregated hyperboxes are allowed to overlap with hyperboxes represented other classes
#                                            + False: no overlap among hyperboxes allowed
#                
#                OUTPUT
#                    V, W, classId, centroid, no_pat are adjusted                                          
#        """
#        yX, xX = self.V.shape
#        time_start = time.perf_counter()
#        # training
#        isTraining = True
#        while isTraining:
#            isTraining = False
#            
#            k = 0 # input pattern index
#            while k < len(self.classId):
#                idx_same_classes = np.logical_or(self.classId == self.classId[k], self.classId == 0)
#                idx_same_classes[k] = False # remove element in the position k
#                idex = np.arange(len(self.classId))
#                idex = idex[idx_same_classes] # keep the indices of elements retained
#                V_same_class = self.V[idx_same_classes]
#                W_same_class = self.W[idx_same_classes]
#                
#                if self.simil_type == 'short':
#                    b = memberG(self.W[k], self.V[k], V_same_class, W_same_class, self.gamma, self.oper)
#                elif self.simil_type == 'long':
#                    b = memberG(self.V[k], self.W[k], W_same_class, V_same_class, self.gamma, self.oper)
#                else:
#                    b = asym_similarity_one_many(self.V[k], self.W[k], V_same_class, W_same_class, self.gamma, self.oper_asym, self.oper)
#                
#                indB = np.argsort(b)[::-1]
#                idex = idex[indB]
#                sortB = b[indB]
#                
#                maxB = sortB[sortB >= self.simil_thres]	# apply membership threshold
#                
#                if len(maxB) > 0:
#                    idexmax = idex[sortB >= self.simil_thres]
#                    
#                    pairewise_maxb = np.concatenate((np.minimum(k, idexmax)[:, np.newaxis], np.maximum(k,idexmax)[:, np.newaxis], maxB[:, np.newaxis]), axis=1)
#
#                    for i in range(pairewise_maxb.shape[0]):
#                        # calculate new coordinates of k-th hyperbox by including pairewise_maxb(i,1)-th box, scrap the latter and leave the rest intact
#                        # agglomorate pairewise_maxb(i, 0) and pairewise_maxb(i, 1) by adjusting pairewise_maxb(i, 0)
#                        # remove pairewise_maxb(i, 1) by getting newV from 1 -> pairewise_maxb(i, 0) - 1, new coordinates for pairewise_maxb(i, 0), from pairewise_maxb(i, 0) + 1 -> pairewise_maxb(i, 1) - 1, pairewise_maxb(i, 1) + 1 -> end
#                        ind_hyperbox_1 = int(pairewise_maxb[i, 0])
#                        ind_hyperbox_2 = int(pairewise_maxb[i, 1])
#                        newV = np.concatenate((self.V[:ind_hyperbox_1], np.minimum(self.V[ind_hyperbox_1], self.V[ind_hyperbox_2]).reshape(1, -1), self.V[ind_hyperbox_1 + 1:ind_hyperbox_2], self.V[ind_hyperbox_2 + 1:]), axis=0)
#                        newW = np.concatenate((self.W[:ind_hyperbox_1], np.maximum(self.W[ind_hyperbox_1], self.W[ind_hyperbox_2]).reshape(1, -1), self.W[ind_hyperbox_1 + 1:ind_hyperbox_2], self.W[ind_hyperbox_2 + 1:]), axis=0)
#                        newClassId = np.concatenate((self.classId[:ind_hyperbox_2], self.classId[ind_hyperbox_2 + 1:]))
#                       
##                        index_remain = np.ones(len(self.classId)).astype(np.bool)
##                        index_remain[ind_hyperbox_2] = False
##                        newV = self.V[index_remain]
##                        newW = self.W[index_remain]
##                        newClassId = self.classId[index_remain]
##                        if ind_hyperbox_1 < ind_hyperbox_2:
##                            tmp_row = ind_hyperbox_1
##                        else:
##                            tmp_row = ind_hyperbox_1 - 1
##                        newV[tmp_row] = np.minimum(self.V[ind_hyperbox_1], self.V[ind_hyperbox_2])
##                        newW[tmp_row] = np.maximum(self.W[ind_hyperbox_1], self.W[ind_hyperbox_2])
##                        
#                        # adjust the hyperbox if no overlap and maximum hyperbox size is not violated
#                        # position of adjustment is pairewise_maxb[i, 0] in new bounds
#                        no_overlap = True
#                        if isAllowedOverlap == False:
#                            no_overlap = not isOverlap(newV, newW, pairewise_maxb[i, 0].astype(np.int64), newClassId)
#                        
#                        if no_overlap and (((newW[pairewise_maxb[i, 0].astype(np.int64)] - newV[pairewise_maxb[i, 0].astype(np.int64)]) <= self.teta_agglo).all() == True):
#                            self.V = newV
#                            self.W = newW
#                            self.classId = newClassId
#                            
#                            # merge centroids and tune the number of patterns contained in the newly aggregated hyperbox, delete data of the eliminated hyperbox
#                            self.centroid[ind_hyperbox_1] = (self.no_pat[ind_hyperbox_1] * self.centroid[ind_hyperbox_1] + self.no_pat[ind_hyperbox_2] * self.centroid[ind_hyperbox_2]) / (self.no_pat[ind_hyperbox_1] + self.no_pat[ind_hyperbox_2])
#                            # delete centroid of hyperbox ind_hyperbox_2
#                            self.centroid = np.concatenate((self.centroid[:ind_hyperbox_2], self.centroid[ind_hyperbox_2 + 1:]), axis=0)
#                            
#                            self.no_pat[ind_hyperbox_1] = self.no_pat[ind_hyperbox_1] + self.no_pat[ind_hyperbox_2]
#                            self.no_pat = np.concatenate((self.no_pat[:ind_hyperbox_2], self.no_pat[ind_hyperbox_2 + 1:]))
#                          
#                            isTraining = True
#                            
#                            if k != pairewise_maxb[i, 0]: # position pairewise_maxb[i, 1] (also k) is removed, so next step should start from pairewise_maxb[i, 1]
#                                k = k - 1
#                                
#                            break # if hyperbox adjusted there's no need to look at other hyperboxes
#                            
#                        
#                k = k + 1
#        
#        self.phase2_elapsed_training_time = time.perf_counter() - time_start
#        print("No. hyperboxes after phase 2: ", len(self.classId))
#        print('Phase 2 running time =', self.phase2_elapsed_training_time)
    
    
    def granular_phase_two_classifier(self, XlT, XuT, patClassIdTest, file_object_save=None):
        """
            Phase 2 in the classifier: using modified online learning to aggregate smaller hyperboxes with the same class
            
                granular_phase_two_classifier(max_hyperbox_size)
                
                INPUT
                    XlT               Test data lower bounds (rows = objects, columns = features)
                    XuT               Test data upper bounds (rows = objects, columns = features)
                    patClassIdTest    Test data class labels (crisp)
                    file_object_save    The file object to write down the results
              
                OUTPUT
                    V, W, classId, centroid, no_pat are adjusted                                          
        """
        nprocs = get_num_cpu_cores() # get number of cores in cpu for handling data
        
        self.phase2_elapsed_training_time = 0
        for teta in self.higher_teta:
            V = []
            if len(self.classId) > 1:
                start_t = time.perf_counter()
                for i in range(len(self.classId)):
                    if len(V) == 0:
                        V = np.array([self.V[i]])
                        W = np.array([self.W[i]])
                        classId = np.array([self.classId[i]])
                        no_pat = np.array([self.no_pat[i]])
                        centroid = np.array([self.centroid[i]])
                    else:
                        classOfX = self.classId[i]
                        id_lb_sameX = np.logical_or(classId == classOfX, classId == 0)
                        
                        isAddNew = False
                        if id_lb_sameX.any() == True: 
                            V_sameX = V[id_lb_sameX]
                            W_sameX = W[id_lb_sameX]
                            lb_sameX = classId[id_lb_sameX]
                            id_range = np.arange(len(classId))
                            id_processing = id_range[id_lb_sameX]
        
                            b = memberG(self.V[i], self.W[i], V_sameX, W_sameX, self.gamma)
                            index = np.argsort(b)[::-1]
                            bSort = b[index]
                        
                            if bSort[0] != 1 or (classOfX != lb_sameX[index[0]] and classOfX != 0):
                                adjust = False
                                maxB = bSort[bSort >= self.simil_thres]	# apply membership threshold
                                
                                if len(maxB) > 0:
                                    indexmax = index[bSort >= self.simil_thres]
                                    for j in id_processing[indexmax]:
                                        # test violation of max hyperbox size and class labels
                                        if (classOfX == classId[j] or classId[j] == 0 or classOfX == 0) and ((np.maximum(W[j], self.W[i]) - np.minimum(V[j], self.V[i])) <= teta).all() == True:
                                            # save old value
                                            Vj_old = V[j].copy()
                                            Wj_old = W[j].copy()
                                            classId_old = classId[j]
                                            
                                            # adjust the j-th hyperbox
                                            V[j] = np.minimum(V[j], self.V[i])
                                            W[j] = np.maximum(W[j], self.W[i])
                                            
                                            if classOfX != 0 and classId[j] == 0:
                                                classId[j] = classOfX
                                            
                                            # Test overlap        
                                            if modifiedIsOverlap(V, W, j, classId) == True:		# overlap test
                                                # revert change and Choose other hyperbox
                                                V[j] = Vj_old
                                                W[j] = Wj_old
                                                classId[j] = classId_old
                                            else:
                                                # Keep changes and update centroid, stopping the process of choosing hyperboxes
                                                no_pat[j] = no_pat[j] + self.no_pat[i]
                                                centroid[j] = centroid[j] + (self.no_pat[i] / no_pat[j]) * (self.centroid[i] - centroid[j])                            
                                                
                                                adjust = True
                                                break
    
                                # if i-th sample did not fit into any existing box, create a new one
                                if not adjust:
                                    isAddNew = True
                                
                        else:
                            isAddNew = True
                            
                        if isAddNew == True:
                            V = np.concatenate((V, self.V[i].reshape(1, -1)), axis = 0)
                            W = np.concatenate((W, self.W[i].reshape(1, -1)), axis = 0)
                            classId = np.concatenate((classId, [classOfX]))
                            no_pat = np.concatenate((no_pat, [self.no_pat[i]]))
                            centroid = np.concatenate((centroid, self.centroid[i].reshape(1, -1)), axis = 0)
                            # Test overlap and do contraction with current hyperbox because phase 1 create overlapping regions
                            indOfWinner = len(classId) - 1
                            for ii in range(V.shape[0]):
                                if ii != indOfWinner and classId[ii] != classId[indOfWinner]:
                                    caseDim = hyperboxOverlapTest(V, W, indOfWinner, ii)		# overlap test
        
                                    if caseDim.size > 0:
                                        V, W = hyperboxContraction(V, W, caseDim, ii, indOfWinner)
                                            
                            
                self.V = V
                self.W = W
                self.classId = classId
                self.no_pat = no_pat
                self.centroid = centroid
                
                sub_space_time = time.perf_counter() - start_t
                self.phase2_elapsed_training_time = self.phase2_elapsed_training_time + sub_space_time
                
                if file_object_save is not None:
                    file_object_save.write("=> teta = %f \n" % teta)
                    file_object_save.write("Num hyperboxes = %d \n" % len(self.classId))
                    file_object_save.write("Running time = %f \n" % sub_space_time)
                # Do testing
                result_testing = self.parallel_predict_test(nprocs, XlT, XuT, patClassIdTest)
                if (result_testing is not None) and (file_object_save is not None):
                    numTestSample = XlT.shape[0]  
                    file_object_save.write("Number of testing samples = %d \n" % numTestSample)
                    file_object_save.write("Number of wrong predicted samples = %d \n" % result_testing.summis)
                    file_object_save.write("Error Rate = %f \n" % (np.round(result_testing.summis / numTestSample * 100, 4)))
                    file_object_save.write("AUC = %f \n" % result_testing.auc)
                    file_object_save.write("No. samples use centroid for prediction = %d \n" % result_testing.use_centroid)
                    file_object_save.write("No. samples use centroid but wrong prediction = %d \n" % result_testing.use_centroid_wrong)
                
                                
        if file_object_save is not None:
            file_object_save.write("Phase 2 training time = %f \n" % self.phase2_elapsed_training_time)
            
            
    def predict(self, Xl, Xu):
        """
            Predict the class of the input layer
                
                result = predict_test(XlT,XuT)
                
                 INPUT
                     Xl         Input data lower bounds (rows = objects, columns = features)
                     Xu         Input data upper bounds (rows = objects, columns = features)
                     
                 OUTPUT
                     result     A list of predicted results for input samples
              
        """
        #initialization
        yX = Xl.shape[0]
        result = np.empty(yX)
        mem = np.zeros((yX, self.V.shape[0]))
        
        # classifications
        for i in range(yX):
            mem[i, :] = memberG(Xl[i, :], Xu[i, :], self.V, self.W, self.gamma, self.oper) # calculate memberships for all hyperboxes
            bmax = mem[i,:].max()	                                          # get max membership value
            maxVind = np.nonzero(mem[i,:] == bmax)[0]                         # get indexes of all hyperboxes with max membership
            
            if len(maxVind) == 1:
                # only one hyperbox with the highest membership value                
                result[i] = self.classId[maxVind[0]]               
            else:
                # More than one hyperbox with the highest membership value => compare with centroid
                same_first_el_class = maxVind[self.classId[maxVind] == self.classId[maxVind[0]]]
                
                if len(maxVind) == len(same_first_el_class):
                    # all membership in maxVind have the same class
                    result[i] = self.classId[maxVind[0]] 
                else:
                    # at least one pair of hyperboxes with different class => compare the centroid, and classify the input to the hyperboxes with nearest distance to the input pattern
                    centroid_input_pat = (Xl[i] + Xu[i]) / 2
                    id_min = maxVind[0]
                    min_dist = np.linalg.norm(self.centroid[id_min] - centroid_input_pat)
    
                    for j in range(1, len(maxVind)):
                        id_j = maxVind[j]
                        dist_j = np.linalg.norm(self.centroid[id_j] - centroid_input_pat)
                        
                        if dist_j < min_dist or (dist_j == min_dist and self.no_pat[id_j] > self.no_pat[id_min]):
                            id_min = id_j
                            min_dist = dist_j
                    
                    result[i] = self.classId[id_min]
        
        return result
    
    
    def parallel_predict_test(self, nprocs, XlT, XuT, patClassIdTest):
        """
        GFMM classification with hyperboxes stored in self. V, W, classId, centroid, no_pat
        For testing process
    
          result = parallel_predict_test(nprocs,XlT,XuT,patClassIdTest)
    
            INPUT
              XlT               Test data lower bounds (rows = objects, columns = features)
              XuT               Test data upper bounds (rows = objects, columns = features)
              patClassIdTest    Test data class labels (crisp)
              nprocs            Number of processes
              
           OUTPUT
              result           A object with Bunch datatype containing all results as follows:
                                  + summis           Number of misclassified objects
                                  + misclass         Binary error map
                                  + mem              Hyperbox memberships
                                  + use_centroid     The number of patterns must use centroid to make prediction
                                  + use_centroid_wrong  The number of patterns uses centroid to mak prediction but still get wrong result
    
        """
        lst_results = []
        futures = []
        
        with ProcessPoolExecutor(max_workers=nprocs) as executor:
            chunksize = int(math.ceil(len(patClassIdTest) / float(nprocs)))
            for i in range(nprocs):
                XlT_i = XlT[(chunksize * i) : (chunksize * (i + 1))]
                XuT_i = XuT[(chunksize * i) : (chunksize * (i + 1))]
                patClassId_i = patClassIdTest[(chunksize * i) : (chunksize * (i + 1))]
                futures.append(executor.submit(self.predict_test, XlT_i, XuT_i, patClassId_i))
            # Instruct workers to process results as they come, when all are completed
            as_completed(futures) # wait all workers completed:
            for future in futures:
                lst_results.append(future.result())
        
        summis = 0
        misclass = np.array([])
        pred_score = np.array([])
        real_class = np.array([])
        mem = np.array([])
        numPatUsingCentroid = 0
        numPatUsingCentroidWrong = 0
        
        for i in range(len(lst_results)):
            summis = summis + lst_results[i].summis
            numPatUsingCentroid = numPatUsingCentroid + lst_results[i].use_centroid
            numPatUsingCentroidWrong = numPatUsingCentroidWrong + lst_results[i].use_centroid_wrong
            if len(mem) == 0:
                mem = np.array(lst_results[i].mem)
            else:
                mem = np.vstack((mem, lst_results[i].mem))
            
            if len(misclass) == 0:
                misclass = np.array(lst_results[i].misclass)
            else:
                misclass = np.append(misclass, lst_results[i].misclass)
                
            if len(pred_score) == 0:
                pred_score = np.array(lst_results[i].pred_score)
            else:
                pred_score = np.append(pred_score, lst_results[i].pred_score)
                
            if len(real_class) == 0:
                real_class = np.array(lst_results[i].real_class)
            else:
                real_class = np.append(real_class, lst_results[i].real_class)
        
        auc = 0
        unique_label = np.unique(self.classId)
        if len(unique_label) == 2:
            numEle_c1 = len(np.nonzero(self.classId == unique_label[0])[0])
            numEle_c2 = len(np.nonzero(self.classId == unique_label[1])[0])
            if numEle_c1 < numEle_c2:
                positive_label = unique_label[0]
            else:
                positive_label = unique_label[1]
                
            fpr, tpr, _ = metrics.roc_curve(real_class, pred_score, pos_label=positive_label)
            auc = metrics.auc(fpr, tpr)
            
        result = Bunch(summis = summis, misclass = misclass, mem = mem, use_centroid=numPatUsingCentroid, use_centroid_wrong=numPatUsingCentroidWrong, auc = auc)
        
        return result
        
    
    def predict_test(self, XlT, XuT, patClassIdTest):
        """
        GFMM classification with hyperboxes stored in self. V, W, classId, centroid, no_pat
        For testing process
    
          result = predict_test(XlT,XuT,patClassIdTest)
    
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
                                  + mem              Hyperbox memberships
                                  + use_centroid     The number of patterns must use centroid to make prediction
                                  + use_centroid_wrong  The number of patterns uses centroid to mak prediction but still get wrong result
    
        """
    
        #initialization
        yX = XlT.shape[0]
        misclass = np.zeros(yX)
        mem = np.zeros((yX, self.V.shape[0]))
        numPatUsingCentroid = 0
        numPatUsingCentroidWrong = 0
        pred_score = np.array([])
        unique_label = np.unique(self.classId)
        isComputeAUC = False
        save_real_class_test = np.array(patClassIdTest)
        if len(unique_label) == 2:
            pred_score = np.zeros(yX)
            isComputeAUC = True
            numEle_c1 = len(np.nonzero(self.classId == unique_label[0])[0])
            numEle_c2 = len(np.nonzero(self.classId == unique_label[1])[0])
            if numEle_c1 < numEle_c2:
                positive_label = unique_label[0]
            else:
                positive_label = unique_label[1]
                
        # classifications
        for i in range(yX):
            mem[i, :] = memberG(XlT[i, :], XuT[i, :], self.V, self.W, self.gamma, self.oper) # calculate memberships for all hyperboxes
            bmax = mem[i,:].max()	                                          # get max membership value
            maxVind = np.nonzero(mem[i,:] == bmax)[0]                         # get indexes of all hyperboxes with max membership
            
            if len(maxVind) == 1:
                # only one hyperbox with the highest membership value
                if self.classId[maxVind[0]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
                    
                if isComputeAUC == True:
                    if self.classId[maxVind[0]] == positive_label:
                        pred_score[i] = bmax
                    else:
                        pred_score[i] = 1 - bmax                  
            
            else:
                # More than one hyperbox with the highest membership value => compare with centroid
                same_first_el_class = maxVind[self.classId[maxVind] == self.classId[maxVind[0]]]
                
                if len(maxVind) == len(same_first_el_class):
                    # all membership in maxVind have the same class
                    if self.classId[maxVind[0]] == patClassIdTest[i]:
                        misclass[i] = False              
                    else:
                        misclass[i] = True
                    
                    if isComputeAUC == True:
                        if self.classId[maxVind[0]] == positive_label:
                            pred_score[i] = bmax
                        else:
                            pred_score[i] = 1 - bmax 
                else:
                    # at least one pair of hyperboxes with different class => compare the centroid
                    numPatUsingCentroid = numPatUsingCentroid + 1
                    centroid_input_pat = (XlT[i] + XuT[i]) / 2
                    id_min = maxVind[0]
                    min_dist = np.linalg.norm(self.centroid[id_min] - centroid_input_pat)
    
                    for j in range(1, len(maxVind)):
                        id_j = maxVind[j]
                        dist_j = np.linalg.norm(self.centroid[id_j] - centroid_input_pat)
                        
                        if dist_j < min_dist or (dist_j == min_dist and self.no_pat[id_j] > self.no_pat[id_min]):
                            id_min = id_j
                            min_dist = dist_j
                            
                    if self.classId[id_min] != patClassIdTest[i] and patClassIdTest[i] != 0:
                        misclass[i] = True
                        numPatUsingCentroidWrong = numPatUsingCentroidWrong + 1
                    else:
                        misclass[i] = False
                    
                    if isComputeAUC == True:
                        if self.classId[id_min] == positive_label:
                            pred_score[i] = 1
                        else:
                            pred_score[i] = 0
        # results
        summis = np.sum(misclass).astype(np.int64)
    
        result = Bunch(summis = summis, misclass = misclass, mem = mem, use_centroid=numPatUsingCentroid, use_centroid_wrong=numPatUsingCentroidWrong, pred_score = pred_score, real_class = save_real_class_test)
        
        return result
    
if __name__ == '__main__': 
    """
    INPUT parameters from command line
        arg1:   path to file containing the training dataset
        arg2:   path to file containing the testing dataset
        arg3:   path to file containing the validation dataset (using "" if no using validation)
        arg4:   The number of samples in each chunk to be read from the training file (default: 1000000)
        arg5:   The type of split data in each chunk
                + 1:            heterogeneous data with different class label
                + otherwise:    data are grouped by class labels
        arg6:   Maximum size of hyperboxes for online learning (teta_oln, default: 0.1)
        arg7:   gamma value (default: 1)
        arg8:   Similarity threshold (default: 0.5)
        arg9:   operation used to compute membership value: 'min' or 'prod' (default: 'min')
        arg10:  Minimum patterns contained in each hyperbox (default: 4)
    """
    # Init default parameters
    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    validation_file = sys.argv[3]
    print(validation_file)
    
    if (not validation_file) == True:
        # empty validation file
        print('no pruning')
        isPruning = False
    else:
        print('pruning')
        isPruning = True
    
    if len(sys.argv) < 5:
        chunk_size = 1000000
    else:
        chunk_size = int(sys.argv[4])
    print('chunk_size =', chunk_size)
    if len(sys.argv) < 6:
        type_chunk = 0
    else:
        type_chunk = int(sys.argv[5])
    
    print('type=', type_chunk)
    
    if len(sys.argv) < 7:
        teta_list = [0.1, 0.5]
    else:
        teta_list = norm_range = ast.literal_eval(sys.argv[6])
    
    print('teta list = ', teta_list)
    
    if len(sys.argv) < 8:
        gamma = 1
    else:
        gamma = float(sys.argv[7])
    
    if len(sys.argv) < 9:
        simil_thres = 0.5
    else:
        simil_thres = float(sys.argv[8])
    
    if len(sys.argv) < 10:
        oper = 'min'
    else:
        oper = sys.argv[9]
        
    if len(sys.argv) < 11:
        accuracyPerBox = 0.5
    else:
        accuracyPerBox = float(sys.argv[10])
    
    core = get_num_cpu_cores()
    
    if type_chunk == 1:
        pathFileSaveData = '../Experiment/NewAlg/New_' + str(core) + '_hete_result_' + os.path.basename(training_file)
    else:
        pathFileSaveData = '../Experiment/NewAlg/New_' + str(core) + '_homo_result_' + os.path.basename(training_file)
    
    # Read testing file
    _, Xtest, _, patClassIdTest = loadDataset(testing_file, 0, False)
    
    file_object_save = open(pathFileSaveData, "w") 
    
    for exe_time in range(10):
        file_object_save.write("\n\n Time %d \n" % (exe_time + 1))
    
        classifier = Info_Presentation_Multi_Layer_Classifier_GFMM(teta = teta_list, gamma = gamma, simil_thres = simil_thres, oper = oper)
        print('Finish first stage')
        classifier.granular_phase_one_classifier(training_file, chunk_size, type_chunk, isPruning, validation_file, accuracyPerBox, Xtest, Xtest, patClassIdTest, file_object_save)
        
        if len(classifier.classId) > 0:
            classifier.granular_phase_two_classifier(Xtest, Xtest, patClassIdTest, file_object_save)          
        else:
            print("All hyperboxes are prunned")
        
    file_object_save.close()
    print("Finish")
        
    
    
        
        
        
        
        
        
        





