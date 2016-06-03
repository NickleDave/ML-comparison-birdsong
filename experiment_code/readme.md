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
Once you have met the above requirements, navigate to the "experiments" directory in the Anaconda Prompt, then type:  
 `>python test_linsvm_svmrbf_and_knn.py`

## Analyzing the results
When the main script finishes running, it will have created databases (using the Python "shelve" module) in the TARGET_RESULTS_DIR. Each database contains results from one replicate for one condition. A separate script iterates through these database files and creates a summary file. Simply run it from the Anaconda prompt:
`>python generate_summary_results_files_linsvm_svmrbf_knn.py`
I generated figures from results with an iPython notebook, that can be found in the [figure_code directory](https://github.com/NickleDave/ML-comparison-birdsong/tree/master/figure_code) of this repository. The .ipynb in that directory loads results encoded in JSON format. This experiment_code directory also contains the script to convert from the summary file format to JSON. Again make sure the constants in the script are defined appropriately, then run from Anaconda prompt:
`>python make_json_file_of_all_results.py`
