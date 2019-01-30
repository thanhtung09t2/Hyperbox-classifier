# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 10:10:14 2018

@author: Thanh Tung Khuat

Analyze the result file and save result as a record line
"""
import re
import numpy as np
import os

def analyzeFileResult(filePath):
    fileSaveName = 'Analyzed_' + os.path.basename(filePath) + '.csv'
    with open(filePath) as f:
        newBegin = False
        numIteration = 0
        indexLevel = -1
        numTestingSamples = 0
        for line in f:
            line = line.strip()
            if 'Time' in line:
                indexLevel = -1
                numIteration = numIteration + 1
                nums = [int(s) for s in line.split() if s.isdigit()]
                iters = nums[0]
                beforePruning = False
                afterPruning = False
                if iters == 1:
                    # Init data
                    newBegin = True
                    phase1TrainingTime = 0
                    phase2TrainingTime = 0
                    runningTimeHigerLevel = np.array([]) # Using addition operator to current value at each iteration, finally compute average value
                    teta = np.array([])
                    numBoxesBeforePruning = 0
                    numBoxesAfterPruning = 0
                    numBoxesHigherLevel = np.array([])
                    errorRateBeforePruning = 0
                    errorRateAfterPruning = 0
                    errorRateHigherLevel = np.array([])
                    numWrongPredictedSampleBeforePruning = 0
                    numWrongPredictedSampleAfterPruning = 0
                    numWrongPredictedSampleHigherLevel = np.array([])
                    numSampleUsingCentroidBeforePruning = 0
                    numSampleUsingCentroidAfterPruning = 0
                    numSampleUsingCentroidWrongBeforePruning = 0
                    numSampleUsingCentroidWrongAfterPruning = 0
                    numSampleUsingCentroidHigherLevel = np.array([]) # Using addition operator to current value at each iteration, finally compute average value
                    numSampleUsingCentroidWrongHigherLevel = np.array([])
                    
                else:
                    newBegin = False
            else:
                dataLine = re.findall(r"[-+]?\d*\.*\d+", line)
                nums = [float(s) for s in dataLine]

                if 'Number of testing samples' in line:
                    numTestingSamples = nums[0]
                elif 'Phase 1 before pruning' in line:
                    beforePruning = True
                elif 'Phase 1 after pruning' in line:
                    beforePruning = False
                    afterPruning = True
                elif 'Phase 1 running time' in line:
                    beforePruning = False
                    afterPruning = False
                    phase1TrainingTime = phase1TrainingTime + nums[1]
                elif 'Number of wrong predicted samples' in line:
                    if beforePruning == True:
                        numWrongPredictedSampleBeforePruning = numWrongPredictedSampleBeforePruning + nums[0]
                    elif afterPruning == True:
                        numWrongPredictedSampleAfterPruning = numWrongPredictedSampleAfterPruning + nums[0]
                    else:
                        if newBegin == True:
                            numWrongPredictedSampleHigherLevel = np.append(numWrongPredictedSampleHigherLevel, nums[0])
                        else:
                            numWrongPredictedSampleHigherLevel[indexLevel] = numWrongPredictedSampleHigherLevel[indexLevel] + nums[0]
                elif 'Error Rate' in line:
                    if beforePruning == True:
                        errorRateBeforePruning = errorRateBeforePruning + nums[0]
                    elif afterPruning == True:
                        errorRateAfterPruning = errorRateAfterPruning + nums[0]
                    else:
                        if newBegin == True:
                            errorRateHigherLevel = np.append(errorRateHigherLevel, nums[0])
                        else:
                            errorRateHigherLevel[indexLevel] = errorRateHigherLevel[indexLevel] + nums[0]
                elif 'No. samples use centroid for prediction' in line:
                    if beforePruning == True:
                        numSampleUsingCentroidBeforePruning = numSampleUsingCentroidBeforePruning + nums[0]
                    elif afterPruning == True:
                        numSampleUsingCentroidAfterPruning = numSampleUsingCentroidAfterPruning + nums[0]
                    else:
                        if newBegin == True:
                            numSampleUsingCentroidHigherLevel = np.append(numSampleUsingCentroidHigherLevel, nums[0])
                        else:
                            numSampleUsingCentroidHigherLevel[indexLevel] = numSampleUsingCentroidHigherLevel[indexLevel] + nums[0]
                elif 'No. samples use centroid but wrong prediction' in line:
                    if beforePruning == True:
                        numSampleUsingCentroidWrongBeforePruning = numSampleUsingCentroidWrongBeforePruning + nums[0]
                    elif afterPruning == True:
                        numSampleUsingCentroidWrongAfterPruning = numSampleUsingCentroidWrongAfterPruning + nums[0]
                    else:
                        if newBegin == True:
                            numSampleUsingCentroidWrongHigherLevel = np.append(numSampleUsingCentroidWrongHigherLevel, nums[0])
                        else:
                            numSampleUsingCentroidWrongHigherLevel[indexLevel] = numSampleUsingCentroidWrongHigherLevel[indexLevel] + nums[0]
                elif 'No. hyperboxes before pruning' in line:
                    numBoxesBeforePruning = numBoxesBeforePruning + nums[0]
                elif 'No. hyperboxes after pruning' in line:
                    numBoxesAfterPruning = numBoxesAfterPruning + nums[0]
                elif 'teta' in line:
                    indexLevel = indexLevel + 1
                    if newBegin == True:
                        teta = np.append(teta, nums[0])
                elif 'Num hyperboxes' in line:
                    if newBegin == True:
                        numBoxesHigherLevel = np.append(numBoxesHigherLevel, nums[0])
                    else:
                        numBoxesHigherLevel[indexLevel] = numBoxesHigherLevel[indexLevel] + nums[0]
                elif 'Running time' in line:
                    if newBegin == True:
                        runningTimeHigerLevel = np.append(runningTimeHigerLevel, nums[0])
                    else:
                        runningTimeHigerLevel[indexLevel] = runningTimeHigerLevel[indexLevel] + nums[0]
                elif 'Phase 2 training time' in line:
                    phase2TrainingTime = phase2TrainingTime + nums[1]
        
        phase1TrainingTime = phase1TrainingTime / numIteration
        phase2TrainingTime = phase2TrainingTime / numIteration
        runningTimeHigerLevel = runningTimeHigerLevel / numIteration
        numBoxesBeforePruning = numBoxesBeforePruning / numIteration
        numBoxesAfterPruning = numBoxesAfterPruning / numIteration      
        numBoxesHigherLevel = numBoxesHigherLevel / numIteration
        errorRateBeforePruning = errorRateBeforePruning / numIteration
        errorRateAfterPruning = errorRateAfterPruning / numIteration
        errorRateHigherLevel = errorRateHigherLevel / numIteration
        numWrongPredictedSampleBeforePruning = numWrongPredictedSampleBeforePruning / numIteration
        numWrongPredictedSampleAfterPruning = numWrongPredictedSampleAfterPruning / numIteration
        numWrongPredictedSampleHigherLevel = numWrongPredictedSampleHigherLevel / numIteration
        numSampleUsingCentroidBeforePruning = numSampleUsingCentroidBeforePruning / numIteration
        numSampleUsingCentroidAfterPruning = numSampleUsingCentroidAfterPruning / numIteration
        numSampleUsingCentroidWrongBeforePruning = numSampleUsingCentroidWrongBeforePruning / numIteration
        numSampleUsingCentroidWrongAfterPruning = numSampleUsingCentroidWrongAfterPruning / numIteration
        numSampleUsingCentroidHigherLevel = numSampleUsingCentroidHigherLevel / numIteration
        numSampleUsingCentroidWrongHigherLevel = numSampleUsingCentroidWrongHigherLevel / numIteration
        
        # print result
        print('Num testing sample = ', numTestingSamples)
        print('phase1TrainingTime = ', phase1TrainingTime)
        print('phase2TrainingTime = ', phase2TrainingTime)
        print('runningTimeHigerLevel = ', runningTimeHigerLevel)
        print('numBoxesBeforePruning = ', numBoxesBeforePruning)
        print('numBoxesAfterPruning = ', numBoxesAfterPruning)
        print('numBoxesHigherLevel = ', numBoxesHigherLevel)
        print('errorRateBeforePruning = ', errorRateBeforePruning)
        print('errorRateAfterPruning = ', errorRateAfterPruning)
        print('errorRateHigherLevel = ', errorRateHigherLevel)
        print('numWrongPredictedSampleBeforePruning = ', numWrongPredictedSampleBeforePruning)
        print('numWrongPredictedSampleAfterPruning = ', numWrongPredictedSampleAfterPruning)
        print('numWrongPredictedSampleHigherLevel = ', numWrongPredictedSampleHigherLevel)
        print('numSampleUsingCentroidBeforePruning = ', numSampleUsingCentroidBeforePruning)
        print('numSampleUsingCentroidAfterPruning = ', numSampleUsingCentroidAfterPruning)
        print('numSampleUsingCentroidWrongBeforePruning = ', numSampleUsingCentroidWrongBeforePruning)
        print('numSampleUsingCentroidWrongAfterPruning = ', numSampleUsingCentroidWrongAfterPruning)
        print('numSampleUsingCentroidHigherLevel = ', numSampleUsingCentroidHigherLevel)
        print('numSampleUsingCentroidWrongHigherLevel = ', numSampleUsingCentroidWrongHigherLevel)
        
        file_object_save = open(fileSaveName, "w")
        file_object_save.write('Num hyperboxes before pruning, Num hyperboxes after pruning, Running time, Error Rate Before Pruning, Error Rate after Pruning, Total sample Using centroid for prediction before Pruning, Total wrong sample Using centroid for prediction before Pruning, Total sample Using centroid for prediction after Pruning, Total wrong sample Using centroid for prediction after Pruning,')
        for i in range(len(teta)):
            file_object_save.write('Num hyperboxes, Running Time, Error Rate, Total sample Using centroid for prediction, Total wrong sample Using centroid for prediction, ')
        file_object_save.writelines('Phase 2 running time total\n')
        
        error_rate_before_pruning = np.round(numWrongPredictedSampleBeforePruning / numTestingSamples * 100, 4)
        error_rate_after_pruning = np.round(numWrongPredictedSampleAfterPruning / numTestingSamples * 100, 4)
        file_object_save.write('%d, %d, %f, %f (%d/%d), %f (%d/%d), %d, %d, %d, %d, ' % (numBoxesBeforePruning, numBoxesAfterPruning, phase1TrainingTime, error_rate_before_pruning, numWrongPredictedSampleBeforePruning, numTestingSamples, error_rate_after_pruning, numWrongPredictedSampleAfterPruning, numTestingSamples, numSampleUsingCentroidBeforePruning, numSampleUsingCentroidWrongBeforePruning, numSampleUsingCentroidAfterPruning, numSampleUsingCentroidWrongAfterPruning))
        for i in range(len(teta)):
            error_rate = np.round(numWrongPredictedSampleHigherLevel[i] / numTestingSamples * 100, 4)
            file_object_save.write('%d, %f, %f (%d/%d), %d, %d, ' % (numBoxesHigherLevel[i], runningTimeHigerLevel[i], error_rate, numWrongPredictedSampleHigherLevel[i], numTestingSamples, numSampleUsingCentroidHigherLevel[i], numSampleUsingCentroidWrongHigherLevel[i]))
            
        file_object_save.writelines('%f\n' % phase2TrainingTime)
        
        file_object_save.close()

# Only use for two-class datasets               
def getOptimalBayesErrorFollowBoundary(dataFilePath, xThresOptimal):
    XC1LargerThanThreshold = 0
    XC1LowerThanThreshold = 0
    XC2LargerThanThreshold = 0
    XC2LowerThanThreshold = 0
    
    with open(dataFilePath) as f:
        for line in f:
            nums = np.fromstring(line.rstrip('\n').replace(',', ' '), dtype=np.float64, sep=' ').tolist()
            if len(nums) > 0:
                if nums[-1] == 1:
                    if nums[0] < xThresOptimal:
                        XC1LowerThanThreshold = XC1LowerThanThreshold + 1
                    else:
                        XC1LargerThanThreshold = XC1LargerThanThreshold + 1
                else:
                    if nums[0] < xThresOptimal:
                        XC2LowerThanThreshold = XC2LowerThanThreshold + 1
                    else:
                        XC2LargerThanThreshold = XC2LargerThanThreshold + 1
    
    total = XC1LargerThanThreshold + XC1LowerThanThreshold + XC2LargerThanThreshold + XC2LowerThanThreshold
    e1 = (XC1LargerThanThreshold + XC2LowerThanThreshold) / total
    e2 = (XC1LowerThanThreshold + XC2LargerThanThreshold) / total
    
    return min(e1, e2)
