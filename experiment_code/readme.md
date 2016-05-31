# Code for experiments in "Comparison of machine learning algorithms applied to birdsong element classification"

## Requirements
* Python 3.5, Numpy, Scipy
 * I used the default installation of Anaconda 2.4.1, 64-bit (AMD64 on Win32), in Windows on a Dell Optiplex 9020
* Scikit-learn
 * from the Anaconda Prompt (command-line window) type:  
 `>pip install scikit-learn`
* Liblinear library
 * I re-compiled it to use the Python API on a 64-bit machine
* Files of features extracted from song of four birds, used to train classifiers
    * [can be dowloaded here](https://drive.google.com/folderview?id=0B0BKW2mh0ySnY3NDcjZCM1dLS1k&usp=drive_web)
* Obviously you'll need all the associated code  
 `>git clone https://github.com//NickleDave/ML-comparison-birdsong`
* The main script,  "test_linsvm_svmrbf_and_knn.py", defines the following folders as constants. You'll need to create them.
 * DATA_DIR = './data_for_testing/'
  * This should contain the files of features downloaded from the link above
 * TARGET_RESULTS_DIR = './linsvm_svmrbf_knn_results_test_script/'
 * JSON_FNAME = './data_for_testing/data_by_bird.JSON'

## Running the experiments
Once you have met the above requirements, navigate to the "experiments" directory in the Anaconda Prompt, then type  
 `>python test_linsvm_svmrbf_and_knn.py`


