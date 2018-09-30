import sys, os
from os.path import dirname
root_path = dirname(dirname(os.getcwd()))
sys.path.insert(0, root_path) # insert root directory to environmental variables

from GFMM.onlinegfmm import OnlineGFMM
import numpy as np
from functionhelper.prepocessinghelper import loadDataset

if __name__ == '__main__':
    
    save_result_folder_path = root_path + '\\Experiment\\Online_Fixed_Teta'
    dataset_path = root_path + '\\Dataset\\train_test'
    
    dataset_names = ['aggregation', 'circle', 'complex9', 'DiagnosticBreastCancer', 'elliptical_10_2', 'fourty', 'glass', 'heart', 'ionosphere', 'iris', 'ringnorm', 'segmentation', 'spherical_5_2', 'spiral', 'synthetic', 'thyroid', 'twonorm', 'waveform', 'wine', 'yeast', 'zelnik6']
    
    for dt in range(len(dataset_names)):
        try:
            training_file = dataset_path + '\\' + dataset_names[dt] + '_train.dat'
            testing_file = dataset_path + '\\' + dataset_names[dt] + '_test.dat'
            
            # Read training file
            Xtr, X_tmp, patClassIdTr, pat_tmp = loadDataset(training_file, 1, False)
            # Read testing file
            X_tmp, Xtest, pat_tmp, patClassIdTest = loadDataset(testing_file, 0, False)
            
            teta_save = np.array([])
            numhyperbox_save = np.array([], dtype=np.int64)
            training_time_save = np.array([])
            testing_error_save = np.array([])
            
            for te in np.arange(0.01, 1, 0.01):
                olnClassifier = OnlineGFMM(gamma = 1, teta = te, tMin = te, isDraw = False, oper = 'min', isNorm = False)
                olnClassifier.fit(Xtr, Xtr, patClassIdTr)
                training_time_save = np.append(training_time_save, olnClassifier.elapsed_training_time)
                teta_save = np.append(teta_save, te)
                numhyperbox_save = np.append(numhyperbox_save, len(olnClassifier.classId))
                
                result = olnClassifier.predict(Xtest, Xtest, patClassIdTest)
                if result != None:
                    numTestSample = Xtest.shape[0]
                    err = result.summis / numTestSample
                    testing_error_save = np.append(testing_error_save, err)
                    
            # save result to file
            data_save = np.hstack((teta_save.reshape(-1, 1), numhyperbox_save.reshape(-1, 1), training_time_save.reshape(-1, 1), testing_error_save.reshape(-1, 1)))
            filename = save_result_folder_path + '\\' + dataset_names[dt] + '.txt'
            
            open(filename, 'w').close() # make existing file empty
            
            with open(filename,'a') as f_handle:
                f_handle.writelines('teta, No hyperboxes, Training time, Testing error\n')
                np.savetxt(f_handle, data_save, fmt='%f', delimiter=', ')
           
        except:
            pass
        
    print('---Finish---')
        



 




