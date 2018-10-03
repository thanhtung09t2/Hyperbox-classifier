# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:21:56 2018

@author: Khuat Thanh Tung
"""

# Run 10 times with different input

import sys, os
from os.path import dirname
root_path = dirname(dirname(os.getcwd()))
sys.path.insert(0, root_path) # insert root directory to environmental variables

from GFMM.decisionlevelensembleclassifier import DecisionLevelEnsembleClassifier
from GFMM.modellevelensembleclassifier import ModelLevelEnsembleClassifier
from GFMM.repeat2fold_decisionlevelensembleclassifier import Repeat2FoldDecisionLevelEnsembleClassifier
from GFMM.repeat2fold_modellevelensembleclassifier import Repeat2FoldModelLevelEnsembleClassifier
import numpy as np
from functionhelper.prepocessinghelper import loadDataset

if __name__ == '__main__':
    
    save_decision_level_result_folder_path = root_path + '\\Experiment\\Ensemble\\Decision_Level'
    save_model_level_result_folder_path = root_path + '\\Experiment\\Ensemble\\Model_Level'
    save_repeate_2fold_decision_result_folder_path = root_path + '\\Experiment\\Ensemble\\Repeat_2fold_Decision'
    save_repeate_2fold_model_result_folder_path = root_path + '\\Experiment\\Ensemble\\Repeat_2fold_Model'
    dataset_path = root_path + '\\Dataset\\train_test'
    
    dataset_names = ['aggregation', 'circle', 'complex9', 'DiagnosticBreastCancer', 'elliptical_10_2', 'fourty', 'glass', 'heart', 'ionosphere', 'iris', 'segmentation', 'spherical_5_2', 'spiral', 'synthetic', 'thyroid', 'wine', 'yeast', 'zelnik6']
    # dataset_names = ['ringnorm', 'twonorm', 'waveform']
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
        # simil_thres = [0.02, 0.1, 0.1, 0.3, 0.4, 0.4, 0.6, 0.7, 0.8, 0.9]
        
        numhyperbox_decision_level_save = np.array([], dtype=np.int64)
        training_time_decision_level_save = np.array([])
        testing_error_decision_save = np.array([])
        
        numhyperbox_model_level_save = np.array([], dtype=np.int64)
        training_time_model_level_save = np.array([])
        testing_error_model_level_save = np.array([])
        
        numhyperbox_repeat_decision_level_save = np.array([], dtype=np.int64)
        training_time_repeat_decision_level_save = np.array([])
        testing_error_repeat_decision_level_save = np.array([])
        
        numhyperbox_repeat_model_level_save = np.array([], dtype=np.int64)
        training_time_repeat_model_level_save = np.array([])
        testing_error_repeat_model_level_save = np.array([])
        
        numTestSample = Xtest.shape[0]
        num_classifiers = 9
        num_folds = 10
        
        for test_time in range(10):
            # Random order of input samples
            pos_rnd = np.random.permutation(Xtr.shape[0])
            Xtr_time_i = Xtr[pos_rnd]
            pathClassIdTr_time_i = patClassIdTr[pos_rnd]
            
            # Do online training
            ensembleDecisionLevel = DecisionLevelEnsembleClassifier(numClassifier = num_classifiers, numFold = num_folds, gamma = 1, teta = teta, bthres = simil_thres, simil = 'short', sing = 'max', oper = 'min', isNorm = False)
            ensembleDecisionLevel.fit(Xtr_time_i, Xtr_time_i, pathClassIdTr_time_i)
            
            training_time_decision_level_save = np.append(training_time_decision_level_save, ensembleDecisionLevel.elapsed_training_time)
            numhyperbox_decision_level_save = np.append(numhyperbox_decision_level_save, ensembleDecisionLevel.numHyperboxes)
            
            result = ensembleDecisionLevel.predict(Xtest, Xtest, patClassIdTest)
            if result != None:
                err = result.summis / numTestSample
                testing_error_decision_save = np.append(testing_error_decision_save, err)
            
            # Do ensemble at model level
            
            ensembleModelLevel = ModelLevelEnsembleClassifier(numClassifier = num_classifiers, numFold = num_folds, gamma = 1, teta = teta, bthres = simil_thres, simil = 'short', sing = 'max', oper = 'min', isNorm = False)
            ensembleModelLevel.fit(Xtr_time_i, Xtr_time_i, pathClassIdTr_time_i)
            
            training_time_model_level_save = np.append(training_time_model_level_save, ensembleModelLevel.elapsed_training_time)
            numhyperbox_model_level_save = np.append(numhyperbox_model_level_save, ensembleModelLevel.numHyperboxes)
            
            result = ensembleModelLevel.predict(Xtest, Xtest, patClassIdTest)
            if result != None:
                err = result.summis / numTestSample
                testing_error_model_level_save = np.append(testing_error_model_level_save, err)
                   
            # Do repeat 2-fold ensemble at decision level
            repeat2foldDecisionLevel = Repeat2FoldDecisionLevelEnsembleClassifier(numClassifier = num_classifiers, gamma = 1, teta = teta, bthres = 0.9, bthres_min = 0.1, simil = 'short', sing = 'max', oper = 'min', isNorm = False)
            repeat2foldDecisionLevel.fit(Xtr_time_i, Xtr_time_i, pathClassIdTr_time_i)
            
            training_time_repeat_decision_level_save = np.append(training_time_repeat_decision_level_save, repeat2foldDecisionLevel.elapsed_training_time)
            numhyperbox_repeat_decision_level_save = np.append(numhyperbox_repeat_decision_level_save, len(repeat2foldDecisionLevel.classId))
            
            result = repeat2foldDecisionLevel.predict(Xtest, Xtest, patClassIdTest)
            if result != None:
                err = result.summis / numTestSample
                testing_error_repeat_decision_level_save = np.append(testing_error_repeat_decision_level_save, err)
            
            # Do repeat 2-fold ensemble at model level
            repeat2foldModelLevel = Repeat2FoldModelLevelEnsembleClassifier(numClassifier = num_classifiers, gamma = 1, teta = teta, bthres = 0.9, bthres_min = 0.1, simil = 'short', sing = 'max', oper = 'min', isNorm = False)
            repeat2foldModelLevel.fit(Xtr_time_i, Xtr_time_i, pathClassIdTr_time_i)
            
            training_time_repeat_model_level_save = np.append(training_time_repeat_model_level_save, repeat2foldModelLevel.elapsed_training_time)
            numhyperbox_repeat_model_level_save = np.append(numhyperbox_repeat_model_level_save, repeat2foldModelLevel.numHyperboxes)
            
            result = repeat2foldModelLevel.predict(Xtest, Xtest, patClassIdTest)
            if result != None:
                err = result.summis / numTestSample
                testing_error_repeat_model_level_save = np.append(testing_error_repeat_model_level_save, err)
                                  
        # save result to file
        data_decision_level_save = np.hstack((numhyperbox_decision_level_save.reshape(-1, 1), training_time_decision_level_save.reshape(-1, 1), testing_error_decision_save.reshape(-1, 1)))
        filename_decision_level = save_decision_level_result_folder_path + '\\' + dataset_names[dt] + '.txt'
        
        open(filename_decision_level, 'w').close() # make existing file empty
        
        with open(filename_decision_level,'a') as f_handle:
            f_handle.write('num_classifiers = %d, num_folds = %d, teta = %f, simil_thres = %f, measure = short\n' % (num_classifiers, num_folds, teta, simil_thres))
            f_handle.writelines('No hyperboxes, Training time, Testing error\n')
            np.savetxt(f_handle, data_decision_level_save, fmt='%f', delimiter=', ')
        
        # Save results of model level ensemble
        data_model_level_save = np.hstack((numhyperbox_model_level_save.reshape(-1, 1), training_time_model_level_save.reshape(-1, 1), testing_error_model_level_save.reshape(-1, 1)))
        filename_model_level = save_model_level_result_folder_path + '\\' + dataset_names[dt] + '.txt'
        
        open(filename_model_level, 'w').close() # make existing file empty
        
        with open(filename_model_level,'a') as f_handle:
            f_handle.write('num_classifiers = %d, num_folds = %d, teta = %f, simil_thres = %f, measure = short\n' % (num_classifiers, num_folds, teta, simil_thres))
            f_handle.writelines('No hyperboxes, Training time, Testing error\n')
            np.savetxt(f_handle, data_model_level_save, fmt='%f', delimiter=', ')
            
        # save results of repeat 2-fold ensembel desicion level
        data_repeate_decision_level_save = np.hstack((numhyperbox_repeat_decision_level_save.reshape(-1, 1), training_time_repeat_decision_level_save.reshape(-1, 1), testing_error_repeat_decision_level_save.reshape(-1, 1)))
        filename_repeate_decision_level = save_repeate_2fold_decision_result_folder_path + '\\' + dataset_names[dt] + '.txt'
        
        open(filename_repeate_decision_level, 'w').close() # make existing file empty
        
        with open(filename_repeate_decision_level,'a') as f_handle:
            f_handle.write('num_classifiers = %d, teta = %f, simil_thres_start = 0.9, simil_thres_end = 0.1, measure = short\n' % (num_classifiers, teta))
            f_handle.writelines('No hyperboxes, Training time, Testing error\n')
            np.savetxt(f_handle, data_repeate_decision_level_save, fmt='%f', delimiter=', ')
        
        # save results of repeat 2-fold ensembel model level
        data_repeate_model_level_save = np.hstack((numhyperbox_repeat_model_level_save.reshape(-1, 1), training_time_repeat_model_level_save.reshape(-1, 1), testing_error_repeat_model_level_save.reshape(-1, 1)))
        filename_repeate_model_level = save_repeate_2fold_model_result_folder_path + '\\' + dataset_names[dt] + '.txt'
        
        open(filename_repeate_model_level, 'w').close() # make existing file empty
        
        with open(filename_repeate_model_level,'a') as f_handle:
            f_handle.write('num_classifiers = %d, teta = %f, simil_thres_start = 0.9, simil_thres_end = 0.1, measure = short\n' % (num_classifiers, teta))
            f_handle.writelines('No hyperboxes, Training time, Testing error\n')
            np.savetxt(f_handle, data_repeate_model_level_save, fmt='%f', delimiter=', ')
           
#        except:
#            pass
        
    print('---Finish---')

