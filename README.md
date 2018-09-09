# Hyperbox-classifier
<b>Implementation of classifiers based on hyper-box representation</b>

Instruction of executing the online version of GFMM (file: onlinegfmm.py):

    python onlinegfmm.py arg1 arg2 arg3 arg4 arg5 arg6 arg7 arg8 arg9 arg10

INPUT parameters from command line:

    arg1:  + 1 - training and testing datasets are located in separated files
           + 2 - training and testing datasets are located in the same files
    arg2:  path to file containing the training dataset (arg1 = 1) or both training and testing datasets (arg1 = 2)
    arg3:  + path to file containing the testing dataset (arg1 = 1)
           + percentage of the training dataset in the input file
    arg4:  + True: drawing hyperboxes during the training process
           + False: no drawing
    arg5:  Maximum size of hyperboxes (teta, default: 1)
    arg6:  The minimum value of maximum size of hyperboxes (teta_min: default = teta)
    arg7:  gamma value (default: 1)
    arg8:  Operation used to compute membership value: 'min' or 'prod' (default: 'min')
    arg9:  Do normalization of datasets or not? True: Normilize, False: No normalize (default: True)
    arg10: range of input values after normalization (default: [0, 1])
    
Example:

    python onlinegfmm.py 1 synthetic_train.dat synthetic_test.dat True 0.6 0.5 1 min True "[0, 1]"
    
![alt text](https://github.com/thanhtung09t2/Hyperbox-classifier/blob/master/Images/Demo.PNG)
    
If using Spyder to run the source code, let configure Spyder as follows:

From <b>Run/Configuration</b> per file or press <b>Ctr+F6</b>, on the open window, check on <b>Command line options</b> and input the input parameters such as: <i>1 synthetic_train.dat synthetic_test.dat True 0.6 0.5 1 min True "[0, 1]"</i>.

To the drawing display on the separated window (not inline mode as default), from <b>Tools</b> choose <b>Preferences</b>, and then select <b>IPython console</b>, in tab <b>Graphics</b> let change the value of the field <b>backends</b> to <b>Qt5</b> or <b>Qt4</b>, choose <b>OK</b>. Finally, restart Spyder to update the changes.
