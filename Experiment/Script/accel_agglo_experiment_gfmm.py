import sys, os
from os.path import dirname
root_path = dirname(dirname(os.getcwd()))
sys.path.insert(0, root_path) # insert root directory to environmental variables

from GFMM.accelbatchgfmm import AccelBatchGFMM
import numpy as np
from functionhelper.prepocessinghelper import loadDataset

if __name__ == '__main__':
    
    save_result_folder_path = root_path + '\\Experiment\\Acel_Agglo'
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
        # simil_thres = [0.02, 0.1, 0.1, 0.3, 0.4, 0.4, 0.6, 0.7, 0.8, 0.9]
        
        simil_save = np.array([])
        numhyperbox_short_si_save = np.array([], dtype=np.int64)
        training_time_short_si_save = np.array([])
        testing_error_short_si_save = np.array([])
        
        numhyperbox_long_si_save = np.array([], dtype=np.int64)
        training_time_long_si_save = np.array([])
        testing_error_long_si_save = np.array([])
        
        numhyperbox_midmax_si_save = np.array([], dtype=np.int64)
        training_time_midmax_si_save = np.array([])
        testing_error_midmax_si_save = np.array([])
        
        numhyperbox_midmin_si_save = np.array([], dtype=np.int64)
        training_time_midmin_si_save = np.array([])
        testing_error_midmin_si_save = np.array([])
        
        for simil_thres in np.arange(0.02, 1, 0.02):
            simil_save = np.append(simil_save, simil_thres)
            accelClassifier = AccelBatchGFMM(gamma = 1, teta = teta, bthres = simil_thres, simil = 'short', sing = 'max', isDraw = False, oper = 'min', isNorm = False)
            accelClassifier.fit(Xtr, Xtr, patClassIdTr)
            
            training_time_short_si_save = np.append(training_time_short_si_save, accelClassifier.elapsed_training_time)
            numhyperbox_short_si_save = np.append(numhyperbox_short_si_save, len(accelClassifier.classId))
            
            result = accelClassifier.predict(Xtest, Xtest, patClassIdTest)
            if result != None:
                numTestSample = Xtest.shape[0]
                err = result.summis / numTestSample
                testing_error_short_si_save = np.append(testing_error_short_si_save, err)
        
        for simil_thres in np.arange(0.02, 1, 0.02):
            accelClassifier = AccelBatchGFMM(gamma = 1, teta = teta, bthres = simil_thres, simil = 'long', sing = 'max', isDraw = False, oper = 'min', isNorm = False)
            accelClassifier.fit(Xtr, Xtr, patClassIdTr)
            
            training_time_long_si_save = np.append(training_time_long_si_save, accelClassifier.elapsed_training_time)
            numhyperbox_long_si_save = np.append(numhyperbox_long_si_save, len(accelClassifier.classId))
            
            result = accelClassifier.predict(Xtest, Xtest, patClassIdTest)
            if result != None:
                numTestSample = Xtest.shape[0]
                err = result.summis / numTestSample
                testing_error_long_si_save = np.append(testing_error_long_si_save, err)
                
        for simil_thres in np.arange(0.02, 1, 0.02):
            accelClassifier = AccelBatchGFMM(gamma = 1, teta = teta, bthres = simil_thres, simil = 'mid', sing = 'max', isDraw = False, oper = 'min', isNorm = False)
            accelClassifier.fit(Xtr, Xtr, patClassIdTr)
            
            training_time_midmax_si_save = np.append(training_time_midmax_si_save, accelClassifier.elapsed_training_time)
            numhyperbox_midmax_si_save = np.append(numhyperbox_midmax_si_save, len(accelClassifier.classId))
            
            result = accelClassifier.predict(Xtest, Xtest, patClassIdTest)
            if result != None:
                numTestSample = Xtest.shape[0]
                err = result.summis / numTestSample
                testing_error_midmax_si_save = np.append(testing_error_midmax_si_save, err)
        
        for simil_thres in np.arange(0.02, 1, 0.02):
            accelClassifier = AccelBatchGFMM(gamma = 1, teta = teta, bthres = simil_thres, simil = 'mid', sing = 'min', isDraw = False, oper = 'min', isNorm = False)
            accelClassifier.fit(Xtr, Xtr, patClassIdTr)
            
            training_time_midmin_si_save = np.append(training_time_midmin_si_save, accelClassifier.elapsed_training_time)
            numhyperbox_midmin_si_save = np.append(numhyperbox_midmin_si_save, len(accelClassifier.classId))
            
            result = accelClassifier.predict(Xtest, Xtest, patClassIdTest)
            if result != None:
                numTestSample = Xtest.shape[0]
                err = result.summis / numTestSample
                testing_error_midmin_si_save = np.append(testing_error_midmin_si_save, err)
                
                
        # save result to file
        data_save = np.hstack((simil_save.reshape(-1, 1), numhyperbox_short_si_save.reshape(-1, 1), training_time_short_si_save.reshape(-1, 1), testing_error_short_si_save.reshape(-1, 1), 
                               numhyperbox_long_si_save.reshape(-1, 1), training_time_long_si_save.reshape(-1, 1), testing_error_long_si_save.reshape(-1, 1),
                               numhyperbox_midmax_si_save.reshape(-1, 1), training_time_midmax_si_save.reshape(-1, 1), testing_error_midmax_si_save.reshape(-1, 1),
                               numhyperbox_midmin_si_save.reshape(-1, 1), training_time_midmin_si_save.reshape(-1, 1), testing_error_midmin_si_save.reshape(-1, 1)))
        filename = save_result_folder_path + '\\' + dataset_names[dt] + '.txt'
        
        open(filename, 'w').close() # make existing file empty
        
        with open(filename,'a') as f_handle:
            f_handle.write('Teta = %d \n' % (teta))
            f_handle.writelines('simi thres, No hyperboxes Short simi, Training time Short simi, Testing error Short simi, No hyperboxes Long simi, Training time Long simi, Testing error Long simi, \
                                No hyperboxes Mid max simi, Training time Mid max simi, Testing error Mid max simi,\
                                No hyperboxes Mid min simi, Training time Mid min simi, Testing error Mid min simi \n')
            np.savetxt(f_handle, data_save, fmt='%f', delimiter=', ')
           
#        except:
#            pass
        
    print('---Finish---')