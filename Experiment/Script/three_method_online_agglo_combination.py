# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:09:47 2018

@author: Thanh Tung Khuat
"""

# Run 10 times with different input

import sys, os
from os.path import dirname
root_path = dirname(dirname(os.getcwd()))
sys.path.insert(0, root_path) # insert root directory to environmental variables

from GFMM.onlineagglogfmm import OnlineAggloGFMM
from GFMM.onlineofflinegfmm import OnlineOfflineGFMM
from GFMM.agglo_onlgfmm import AggloOnlineGFMM
from GFMM.accelbatchgfmm import AccelBatchGFMM
from GFMM.onlinegfmm import OnlineGFMM
import numpy as np
from functionhelper.prepocessinghelper import loadDataset, splitDatasetRndTo2Part

if __name__ == '__main__':
    
    save_online_result_folder_path = root_path + '\\Experiment\\Online_Agglo_Combination\\Online'
    save_agglo_result_folder_path = root_path + '\\Experiment\\Online_Agglo_Combination\\Agglo'
    save_online_agglo_result_folder_path = root_path + '\\Experiment\\Online_Agglo_Combination\\Online_Agglo'
    save_agglo_online_result_folder_path = root_path + '\\Experiment\\Online_Agglo_Combination\\Agglo_Online'
    save_parallel_result_folder_path = root_path + '\\Experiment\\Online_Agglo_Combination\\Parallel_Comb'
    dataset_path = root_path + '\\Dataset\\train_test'
    
    # dataset_names = ['circle', 'complex9', 'DiagnosticBreastCancer', 'glass', 'heart', 'ionosphere', 'iris', 'segmentation', 'spherical_5_2', 'spiral', 'synthetic', 'thyroid', 'wine', 'yeast', 'zelnik6']
    dataset_names = ['ringnorm', 'twonorm', 'waveform']
    for dt in range(len(dataset_names)):
        #try:
        print('Current dataset: ', dataset_names[dt])
        training_file = dataset_path + '\\' + dataset_names[dt] + '_train.dat'
        testing_file = dataset_path + '\\' + dataset_names[dt] + '_test.dat'
        
        # Read training file
        Xtr, X_tmp, patClassIdTr, pat_tmp = loadDataset(training_file, 1, False)
        # Read testing file
        X_tmp, Xtest, pat_tmp, patClassIdTest = loadDataset(testing_file, 0, False)
        
        teta = 0.4
        simil_thres = 0.5
        
        numhyperbox_online_save = np.array([], dtype=np.int64)
        training_time_online_save = np.array([])
        testing_error_online_save = np.array([])
        
        numhyperbox_agglo_save = np.array([], dtype=np.int64)
        training_time_agglo_save = np.array([])
        testing_error_agglo_save = np.array([])
        
        numhyperbox_online_agglo_save = np.array([], dtype=np.int64)
        training_time_online_agglo_save = np.array([])
        testing_error_online_agglo_save = np.array([])
        
        numhyperbox_agglo_online_save = np.array([], dtype=np.int64)
        training_time_agglo_online_save = np.array([])
        testing_error_agglo_online_save = np.array([])
        
        numhyperbox_parallel_save = np.array([], dtype=np.int64)
        training_time_parallel_save = np.array([])
        testing_error_parallel_save = np.array([])
        
        # Create full dataset by merging training and testing set, after that spliting again
        # Training: 80%, testing: 20%
        X_full = np.vstack((Xtr, Xtest))
        classId_full = np.append(patClassIdTr, patClassIdTest)
        pivot_pos = int(X_full.shape[0] * 0.8)
        
        for test_time in range(10):
            pos_rnd = np.random.permutation(X_full.shape[0])
            X_full_tmp = X_full[pos_rnd]
            classId_full_tmp = classId_full[pos_rnd]
            
            Xtr_time_i = X_full_tmp[:pivot_pos]
            pathClassIdTr_time_i = classId_full_tmp[:pivot_pos]
            
            Xtest = X_full_tmp[pivot_pos:]
            patClassIdTest = classId_full_tmp[pivot_pos:]
            
            numTestSample = Xtest.shape[0]
            
            # Do online learning
            olnClassifier = OnlineGFMM(gamma = 1, teta = teta, tMin = teta, isDraw = False, oper = 'min', isNorm = False)
            olnClassifier.fit(Xtr_time_i, Xtr_time_i, pathClassIdTr_time_i)
            
            training_time_online_save = np.append(training_time_online_save, olnClassifier.elapsed_training_time)
            numhyperbox_online_save = np.append(numhyperbox_online_save, len(olnClassifier.classId))
            
            result = olnClassifier.predict(Xtest, Xtest, patClassIdTest)
            if result != None:
                err = result.summis / numTestSample
                testing_error_online_save = np.append(testing_error_online_save, err)           
            
            #  Do accelerated learning
            accelClassifier = AccelBatchGFMM(gamma = 1, teta = teta, bthres = simil_thres, simil = 'short', sing = 'max', isDraw = False, oper = 'min', isNorm = False)
            accelClassifier.fit(Xtr_time_i, Xtr_time_i, pathClassIdTr_time_i)
            
            training_time_agglo_save = np.append(training_time_agglo_save, accelClassifier.elapsed_training_time)
            numhyperbox_agglo_save = np.append(numhyperbox_agglo_save, len(accelClassifier.classId))
            
            result = accelClassifier.predict(Xtest, Xtest, patClassIdTest)
            if result != None:
                err = result.summis / numTestSample
                testing_error_agglo_save = np.append(testing_error_agglo_save, err)              
            
            # Do online training before agglo
            olnAggloClassifier = OnlineAggloGFMM(gamma = 1, teta_onl = 0.1, teta_agglo = teta, bthres = simil_thres, simil = 'mid', sing = 'max', isDraw = False, oper = 'min', isNorm = False)
            olnAggloClassifier.fit(Xtr_time_i, Xtr_time_i, pathClassIdTr_time_i)
            
            training_time_online_agglo_save = np.append(training_time_online_agglo_save, olnAggloClassifier.elapsed_training_time)
            numhyperbox_online_agglo_save = np.append(numhyperbox_online_agglo_save, len(olnAggloClassifier.classId))
            
            result = olnAggloClassifier.predict(Xtest, Xtest, patClassIdTest)
            if result != None:
                err = result.summis / numTestSample
                testing_error_online_agglo_save = np.append(testing_error_online_agglo_save, err)
            
            # Do accelerated agglomerative training first, then do online training with different dataset
            Xtr_time_i_onl, Xtr_time_i_off = splitDatasetRndTo2Part(Xtr_time_i, Xtr_time_i, pathClassIdTr_time_i, 0.6)
            
            aggloOnlineClassifier = AggloOnlineGFMM(gamma = 1, teta_onl = teta, teta_agglo = teta, bthres = simil_thres, simil = 'short', sing = 'max', isDraw = False, oper = 'min', isNorm = False)
            aggloOnlineClassifier.fit(Xtr_time_i_onl.lower, Xtr_time_i_onl.upper, Xtr_time_i_onl.label, Xtr_time_i_off.lower, Xtr_time_i_off.upper, Xtr_time_i_off.label, typeOfAgglo = 1)
            
            training_time_agglo_online_save = np.append(training_time_agglo_online_save, aggloOnlineClassifier.elapsed_training_time)
            numhyperbox_agglo_online_save = np.append(numhyperbox_agglo_online_save, len(aggloOnlineClassifier.classId))
            
            result = aggloOnlineClassifier.predict(Xtest, Xtest, patClassIdTest)
            if result != None:
                err = result.summis / numTestSample
                testing_error_agglo_online_save = np.append(testing_error_agglo_online_save, err)
                   
            # Do parallel combination training
            parallelCombClassifier = OnlineOfflineGFMM(gamma = 1, teta_onl = teta, teta_agglo = teta, bthres = simil_thres, simil = 'short', sing = 'max', isDraw = False, oper = 'min', isNorm = False)
            parallelCombClassifier.fit(Xtr_time_i_onl.lower, Xtr_time_i_onl.upper, Xtr_time_i_onl.label, Xtr_time_i_off.lower, Xtr_time_i_off.upper, Xtr_time_i_off.label)
            
            training_time_parallel_save = np.append(training_time_parallel_save, parallelCombClassifier.elapsed_training_time)
            numhyperbox_parallel_save = np.append(numhyperbox_parallel_save, (len(parallelCombClassifier.onlClassifier.classId) + len(parallelCombClassifier.offClassifier.classId)))
            
            result = parallelCombClassifier.predict(Xtest, Xtest, patClassIdTest)
            if result != None:
                err = result.summis / numTestSample
                testing_error_parallel_save = np.append(testing_error_parallel_save, err)
                 
        # save result to file
        data_online_save = np.hstack((numhyperbox_online_save.reshape(-1, 1), training_time_online_save.reshape(-1, 1), testing_error_online_save.reshape(-1, 1)))
        filename_online = save_online_result_folder_path + '\\' + dataset_names[dt] + '.csv'
        
        open(filename_online, 'w').close() # make existing file empty
        
        with open(filename_online,'a') as f_handle:
            f_handle.write('teta = %f\n' % teta)
            f_handle.writelines('No hyperboxes, Training time, Testing error\n')
            np.savetxt(f_handle, data_online_save, fmt='%f', delimiter=', ')
        
        data_agglo_save = np.hstack((numhyperbox_agglo_save.reshape(-1, 1), training_time_agglo_save.reshape(-1, 1), testing_error_agglo_save.reshape(-1, 1)))
        filename_agglo = save_agglo_result_folder_path + '\\' + dataset_names[dt] + '.csv'
        
        open(filename_agglo, 'w').close() # make existing file empty
        
        with open(filename_agglo,'a') as f_handle:
            f_handle.write('teta = %f, short\n' % teta)
            f_handle.writelines('No hyperboxes, Training time, Testing error\n')
            np.savetxt(f_handle, data_agglo_save, fmt='%f', delimiter=', ')
        
        data_online_agglo_save = np.hstack((numhyperbox_online_agglo_save.reshape(-1, 1), training_time_online_agglo_save.reshape(-1, 1), testing_error_online_agglo_save.reshape(-1, 1)))
        filename_online_agglo = save_online_agglo_result_folder_path + '\\' + dataset_names[dt] + '.csv'
        
        open(filename_online_agglo, 'w').close() # make existing file empty
        
        with open(filename_online_agglo,'a') as f_handle:
            f_handle.write('teta_online = 0.1, teta_agglo = %f \n' % teta)
            f_handle.writelines('No hyperboxes, Training time, Testing error\n')
            np.savetxt(f_handle, data_online_agglo_save, fmt='%f', delimiter=', ')
        
        # Save results of accelerated batch learning
        data_agglo_online_save = np.hstack((numhyperbox_agglo_online_save.reshape(-1, 1), training_time_agglo_online_save.reshape(-1, 1), testing_error_agglo_online_save.reshape(-1, 1)))
        filename_agglo_online = save_agglo_online_result_folder_path + '\\' + dataset_names[dt] + '.csv'
        
        open(filename_agglo_online, 'w').close() # make existing file empty
        
        with open(filename_agglo_online,'a') as f_handle:
            f_handle.write('teta_online = teta_agglo = %f, simil_thres = %f, measure = short, online_training = 0.6 * fulldataset \n' % (teta, simil_thres))
            f_handle.writelines('No hyperboxes, Training time, Testing error\n')
            np.savetxt(f_handle, data_agglo_online_save, fmt='%f', delimiter=', ')
            
        # save results of full batch learning
        data_parallel_save = np.hstack((numhyperbox_parallel_save.reshape(-1, 1), training_time_parallel_save.reshape(-1, 1), testing_error_parallel_save.reshape(-1, 1)))
        filename_parallel = save_parallel_result_folder_path + '\\' + dataset_names[dt] + '.csv'
        
        open(filename_parallel, 'w').close() # make existing file empty
        
        with open(filename_parallel,'a') as f_handle:
            f_handle.write('teta_online = teta_agglo = %f, simil_thres = %f, measure = short, online_training = 0.6 * fulldataset \n' % (teta, simil_thres))
            f_handle.writelines('No hyperboxes, Training time, Testing error\n')
            np.savetxt(f_handle, data_parallel_save, fmt='%f', delimiter=', ')
           
#        except:
#            pass
        
    print('---Finish---')

