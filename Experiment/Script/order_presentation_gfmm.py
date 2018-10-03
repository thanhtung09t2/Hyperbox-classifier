# Run 10 times with different input

import sys, os
from os.path import dirname
root_path = dirname(dirname(os.getcwd()))
sys.path.insert(0, root_path) # insert root directory to environmental variables

from GFMM.accelbatchgfmm import AccelBatchGFMM
from GFMM.onlinegfmm import OnlineGFMM
from GFMM.batchgfmm_v1 import BatchGFMMV1
import numpy as np
from functionhelper.prepocessinghelper import loadDataset

if __name__ == '__main__':
    
    save_online_result_folder_path = root_path + '\\Experiment\\Order_Presentation\\Online'
    save_accel_agglo_result_folder_path = root_path + '\\Experiment\\Order_Presentation\\Accel_Agglo'
    save_batch_agglo_result_folder_path = root_path + '\\Experiment\\Order_Presentation\\Batch_Agglo'
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
        
        numhyperbox_online_save = np.array([], dtype=np.int64)
        training_time_online_save = np.array([])
        testing_error_online_save = np.array([])
        
        numhyperbox_accel_agglo_save = np.array([], dtype=np.int64)
        training_time_accel_agglo_save = np.array([])
        testing_error_accel_agglo_save = np.array([])
        
        numhyperbox_batch_agglo_save = np.array([], dtype=np.int64)
        training_time_batch_agglo_save = np.array([])
        testing_error_batch_agglo_save = np.array([])
        
        numTestSample = Xtest.shape[0]
        
        for test_time in range(10):
            # Random order of input samples
            pos_rnd = np.random.permutation(Xtr.shape[0])
            Xtr_time_i = Xtr[pos_rnd]
            pathClassIdTr_time_i = patClassIdTr[pos_rnd]
            
            # Do online training
            olnClassifier = OnlineGFMM(gamma = 1, teta = teta, tMin = teta, isDraw = False, oper = 'min', isNorm = False)
            olnClassifier.fit(Xtr_time_i, Xtr_time_i, pathClassIdTr_time_i)
            
            training_time_online_save = np.append(training_time_online_save, olnClassifier.elapsed_training_time)
            numhyperbox_online_save = np.append(numhyperbox_online_save, len(olnClassifier.classId))
            
            result = olnClassifier.predict(Xtest, Xtest, patClassIdTest)
            if result != None:
                err = result.summis / numTestSample
                testing_error_online_save = np.append(testing_error_online_save, err)
            
            # Do accelerated agglomerative training
            
            accelClassifier = AccelBatchGFMM(gamma = 1, teta = teta, bthres = simil_thres, simil = 'short', sing = 'max', isDraw = False, oper = 'min', isNorm = False)
            accelClassifier.fit(Xtr_time_i, Xtr_time_i, pathClassIdTr_time_i)
            
            training_time_accel_agglo_save = np.append(training_time_accel_agglo_save, accelClassifier.elapsed_training_time)
            numhyperbox_accel_agglo_save = np.append(numhyperbox_accel_agglo_save, len(accelClassifier.classId))
            
            result = accelClassifier.predict(Xtest, Xtest, patClassIdTest)
            if result != None:
                err = result.summis / numTestSample
                testing_error_accel_agglo_save = np.append(testing_error_accel_agglo_save, err)
                   
            # Do full batch training
            batchv1Classifier = BatchGFMMV1(gamma = 1, teta = teta, bthres = simil_thres, simil = 'short', sing = 'max', isDraw = False, oper = 'min', isNorm = False)
            batchv1Classifier.fit(Xtr_time_i, Xtr_time_i, pathClassIdTr_time_i)
            
            training_time_batch_agglo_save = np.append(training_time_batch_agglo_save, batchv1Classifier.elapsed_training_time)
            numhyperbox_batch_agglo_save = np.append(numhyperbox_batch_agglo_save, len(batchv1Classifier.classId))
            
            result = batchv1Classifier.predict(Xtest, Xtest, patClassIdTest)
            if result != None:
                err = result.summis / numTestSample
                testing_error_batch_agglo_save = np.append(testing_error_batch_agglo_save, err)
                 
        # save result to file
        data_online_save = np.hstack((numhyperbox_online_save.reshape(-1, 1), training_time_online_save.reshape(-1, 1), testing_error_online_save.reshape(-1, 1)))
        filename_online = save_online_result_folder_path + '\\' + dataset_names[dt] + '.txt'
        
        open(filename_online, 'w').close() # make existing file empty
        
        with open(filename_online,'a') as f_handle:
            f_handle.writelines('No hyperboxes, Training time, Testing error\n')
            np.savetxt(f_handle, data_online_save, fmt='%f', delimiter=', ')
        
        # Save results of accelerated batch learning
        data_accel_agglo_save = np.hstack((numhyperbox_accel_agglo_save.reshape(-1, 1), training_time_accel_agglo_save.reshape(-1, 1), testing_error_accel_agglo_save.reshape(-1, 1)))
        filename_accel_agglo = save_accel_agglo_result_folder_path + '\\' + dataset_names[dt] + '.txt'
        
        open(filename_accel_agglo, 'w').close() # make existing file empty
        
        with open(filename_accel_agglo,'a') as f_handle:
            f_handle.writelines('No hyperboxes, Training time, Testing error\n')
            np.savetxt(f_handle, data_accel_agglo_save, fmt='%f', delimiter=', ')
            
        # save results of full batch learning
        data_batch_agglo_save = np.hstack((numhyperbox_batch_agglo_save.reshape(-1, 1), training_time_batch_agglo_save.reshape(-1, 1), testing_error_batch_agglo_save.reshape(-1, 1)))
        filename_batch_agglo = save_batch_agglo_result_folder_path + '\\' + dataset_names[dt] + '.txt'
        
        open(filename_batch_agglo, 'w').close() # make existing file empty
        
        with open(filename_batch_agglo,'a') as f_handle:
            f_handle.writelines('No hyperboxes, Training time, Testing error\n')
            np.savetxt(f_handle, data_batch_agglo_save, fmt='%f', delimiter=', ')
           
#        except:
#            pass
        
    print('---Finish---')