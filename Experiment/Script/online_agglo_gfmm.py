# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 10:29:45 2018

@author: Thanh Tung Khuat
"""

import sys, os
from os.path import dirname
root_path = dirname(dirname(os.getcwd()))
sys.path.insert(0, root_path) # insert root directory to environmental variables

from GFMM.onlineagglogfmm import OnlineAggloGFMM
import numpy as np
from functionhelper.prepocessinghelper import loadDataset

if __name__ == '__main__':
    
    save_result_accel_folder_path = root_path + '\\Experiment\\Online_Accel_Agglo'
    save_result_batch_folder_path = root_path + '\\Experiment\\Online_Batch_Agglo'
    dataset_path = root_path + '\\Dataset\\train_test'
    
    # dataset_names = ['aggregation', 'circle', 'complex9', 'DiagnosticBreastCancer', 'elliptical_10_2', 'fourty', 'glass', 'heart', 'ionosphere', 'iris', 'segmentation', 'spherical_5_2', 'spiral', 'synthetic', 'thyroid', 'wine', 'yeast', 'zelnik6']
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
        
        teta_onlines = [0.05, 0.1, 0.15, 0.2, 0.25]
        teta_agglos = [0.5, 0.6, 0.7, 0.8, 0.9]
        # simil_thres = [0.02, 0.1, 0.1, 0.3, 0.4, 0.4, 0.6, 0.7, 0.8, 0.9]
        
        teta_onl_save = np.array([])
        teta_agglo_save = np.array([])
        numhyperbox_online_save = np.array([], dtype=np.int64)
        training_time_save = np.array([])
        testing_error_final_save = np.array([])      
        numhyperbox_final_save = np.array([], dtype=np.int64)
        bthres = 0.5
        simil = 'short'
        for teta_onl in teta_onlines:
            for teta_off in teta_agglos:
                teta_onl_save = np.append(teta_onl_save, teta_onl)
                teta_agglo_save = np.append(teta_agglo_save, teta_off)
                classifier = OnlineAggloGFMM(1, teta_onl, teta_off, bthres, simil, 'max', False, 'min', False)
                classifier.fit(Xtr, Xtr, patClassIdTr, 1) # Accelerated agglomerative learning
                
                training_time_save = np.append(training_time_save, classifier.elapsed_training_time)
                numhyperbox_online_save = np.append(numhyperbox_online_save, classifier.num_hyperbox_after_online)
                numhyperbox_final_save = np.append(numhyperbox_final_save, classifier.num_hyperbox_after_agglo)
                
                result = classifier.predict(Xtest, Xtest, patClassIdTest)
                if result != None:
                    numTestSample = Xtest.shape[0]
                    err = result.summis / numTestSample
                    testing_error_final_save = np.append(testing_error_final_save, err)
            
        
        # save result to file
        data_save = np.hstack((teta_onl_save.reshape(-1, 1), teta_agglo_save.reshape(-1, 1), numhyperbox_online_save.reshape(-1, 1), numhyperbox_final_save.reshape(-1, 1), 
                               training_time_save.reshape(-1, 1), testing_error_final_save.reshape(-1, 1)))
        
        filename = save_result_accel_folder_path + '\\' + dataset_names[dt] + '.txt'
        
        open(filename, 'w').close() # make existing file empty
        
        with open(filename,'a') as f_handle:
            f_handle.write('bthres = %f, simil measure = short\n' % (bthres))
            f_handle.writelines('teta online, teta agglo, Num hyperboxes online, Num hyperbox final, Training time, Testing error')
            np.savetxt(f_handle, data_save, fmt='%f', delimiter=', ')
        
        # For full batch learning faster version
        teta_onl_save = np.array([])
        teta_agglo_save = np.array([])
        numhyperbox_online_save = np.array([], dtype=np.int64)
        training_time_save = np.array([])
        testing_error_final_save = np.array([])      
        numhyperbox_final_save = np.array([], dtype=np.int64)
        bthres = 0.5
        simil = 'short'
        for teta_onl in teta_onlines:
            for teta_off in teta_agglos:
                teta_onl_save = np.append(teta_onl_save, teta_onl)
                teta_agglo_save = np.append(teta_agglo_save, teta_off)
                classifier = OnlineAggloGFMM(1, teta_onl, teta_off, bthres, simil, 'max', False, 'min', False)
                classifier.fit(Xtr, Xtr, patClassIdTr, 0) # Full batch agglomerative learning
                
                training_time_save = np.append(training_time_save, classifier.elapsed_training_time)
                numhyperbox_online_save = np.append(numhyperbox_online_save, classifier.num_hyperbox_after_online)
                numhyperbox_final_save = np.append(numhyperbox_final_save, classifier.num_hyperbox_after_agglo)
                
                result = classifier.predict(Xtest, Xtest, patClassIdTest)
                if result != None:
                    numTestSample = Xtest.shape[0]
                    err = result.summis / numTestSample
                    testing_error_final_save = np.append(testing_error_final_save, err)
            
        
        # save result to file
        data_save = np.hstack((teta_onl_save.reshape(-1, 1), teta_agglo_save.reshape(-1, 1), numhyperbox_online_save.reshape(-1, 1), numhyperbox_final_save.reshape(-1, 1), 
                               training_time_save.reshape(-1, 1), testing_error_final_save.reshape(-1, 1)))
        
        filename = save_result_batch_folder_path + '\\' + dataset_names[dt] + '.txt'
        
        open(filename, 'w').close() # make existing file empty
        
        with open(filename,'a') as f_handle:
            f_handle.write('bthres = %f, simil measure = short\n' % (bthres))
            f_handle.writelines('teta online, teta agglo, Num hyperboxes online, Num hyperbox final, Training time, Testing error')
            np.savetxt(f_handle, data_save, fmt='%f', delimiter=', ')
            
#        except:
#            pass
        
    print('---Finish---')

