#loads results for plotting in Jupyter notebook
#from standard library
import sys
import json
import shelve

#from Anaconda
import numpy as np
import matplotlib.pyplot as plt

#load data from site
sys.path.append('C:/Users/dnicho4/Documents/IPython Notebooks/test_everything/linsvm_svmrbf_knn_results/')

#constants    
DATA_DIR = '../data_for_testing/'
JSON_FNAME = '../data_for_testing/data_by_bird.JSON'
RESULTS_DIR = '../linsvm_svmrbf_knn_results/'
SUMMARY_SHV_BASE_FNAME = '_linsvm_svmrbf_knn_summary_results'

BIRD_NAMES_DICT = {
    'bird 1':'gr41rd51',
    'bird 2':'gy6or6',
    'bird 3':'or60yw70',
    'bird 4':'bl26lb16'
    }
BIRD_NAME_LIST = ['bird 1','bird 2','bird 3','bird 4']

NUM_SONGS_TO_TEST = list(range(3,16,3)) + [21,27,33,39]
# i.e., [3,6,9,...15,21,27,33,39].
REPLICATES = range(1,6)

#load summary data
with open(JSON_FNAME) as json_file:
    data_by_bird = json.load(json_file)

results_dict = {}

for bird_ID in BIRD_NAMES_DICT.values():
    results_fname = RESULTS_DIR + bird_ID + SUMMARY_SHV_BASE_FNAME

    with shelve.open(results_fname) as shv:
        results_dict[bird_ID] = {
            'num_train_samples':shv['num_train_samples'],
            
            'svm_Tach_holdout_rnd_acc':shv['svm_Tach_holdout_rnd_acc'],
            'svm_Tach_holdout_rnd_acc_mn':shv['svm_Tach_holdout_rnd_acc_mn'],
            'svm_Tach_holdout_rnd_acc_std':shv['svm_Tach_holdout_rnd_acc_std'],            
            
            'svm_Tach_test_rnd_acc':shv['svm_Tach_test_rnd_acc'],
            'svm_Tach_test_rnd_acc_mn':shv['svm_Tach_test_rnd_acc_mn'],
            'svm_Tach_test_rnd_acc_std':shv['svm_Tach_test_rnd_acc_std'],
            
            'svm_holdout_rnd_acc':shv['svm_holdout_rnd_acc'],
            'svm_holdout_rnd_acc_mn':shv['svm_holdout_rnd_acc_mn'],
            'svm_holdout_rnd_acc_std':shv['svm_holdout_rnd_acc_std'],

            'svm_test_rnd_acc':shv['svm_test_rnd_acc'],
            'svm_test_rnd_acc_mn':shv['svm_test_rnd_acc_mn'],
            'svm_test_rnd_acc_std':shv['svm_test_rnd_acc_std'],

            'knn_holdout_rnd_acc':shv['knn_holdout_rnd_acc'],
            'knn_holdout_rnd_acc_mn':shv['knn_holdout_rnd_acc_mn'],
            'knn_holdout_rnd_acc_std':shv['knn_holdout_rnd_acc_std'],
            
            'knn_test_rnd_acc':shv['knn_test_rnd_acc'],
            'knn_test_rnd_acc_mn':shv['knn_test_rnd_acc_mn'],
            'knn_test_rnd_acc_std':shv['knn_test_rnd_acc_std'],

            'linsvm_holdout_acc_by_label':shv['linsvm_holdout_acc_by_label'],
            'linsvm_holdout_avg_acc':shv['linsvm_holdout_avg_acc'],
            'linsvm_holdout_avg_acc_mn':shv['linsvm_holdout_avg_acc_mn'],
            'linsvm_holdout_avg_acc_std':shv['linsvm_holdout_avg_acc_std'],
            'linsvm_holdout_cm_arr':shv['linsvm_holdout_cm_arr'],

            'linsvm_test_acc_by_label':shv['linsvm_test_acc_by_label'],
            'linsvm_test_avg_acc':shv['linsvm_test_avg_acc'],
            'linsvm_test_avg_acc_mn':shv['linsvm_test_avg_acc_mn'],
            'linsvm_test_avg_acc_std':shv['linsvm_test_avg_acc_std'],
            'linsvm_test_cm_arr':shv['linsvm_test_cm_arr'],

            'linsvm_holdout_no_intro_acc_by_label':shv['linsvm_holdout_no_intro_acc_by_label'],
            'linsvm_holdout_no_intro_avg_acc':shv['linsvm_holdout_no_intro_avg_acc'],
            'linsvm_holdout_no_intro_avg_acc_mn':shv['linsvm_holdout_no_intro_avg_acc_mn'],
            'linsvm_holdout_no_intro_avg_acc_std':shv['linsvm_holdout_no_intro_avg_acc_std'],
            'linsvm_holdout_no_intro_cm_arr':shv['linsvm_holdout_no_intro_cm_arr'],

            'linsvm_test_no_intro_acc_by_label':shv['linsvm_test_no_intro_acc_by_label'],
            'linsvm_test_no_intro_avg_acc':shv['linsvm_test_no_intro_avg_acc'],
            'linsvm_test_no_intro_avg_acc_mn':shv['linsvm_test_no_intro_avg_acc_mn'],
            'linsvm_test_no_intro_avg_acc_std':shv['linsvm_test_no_intro_avg_acc_std'],
            'linsvm_test_no_intro_cm_arr':shv['linsvm_test_no_intro_cm_arr'],
            
            'svm_Tach_holdout_acc_by_label':shv['svm_Tach_holdout_acc_by_label'],
            'svm_Tach_holdout_avg_acc':shv['svm_Tach_holdout_avg_acc'],
            'svm_Tach_holdout_avg_acc_mn':shv['svm_Tach_holdout_avg_acc_mn'],
            'svm_Tach_holdout_avg_acc_std':shv['svm_Tach_holdout_avg_acc_std'],
            'svm_Tach_holdout_cm_arr':shv['svm_Tach_holdout_cm_arr'],

            'svm_Tach_test_acc_by_label':shv['svm_Tach_test_acc_by_label'],
            'svm_Tach_test_avg_acc':shv['svm_Tach_test_avg_acc'],
            'svm_Tach_test_avg_acc_mn':shv['svm_Tach_test_avg_acc_mn'],
            'svm_Tach_test_avg_acc_std':shv['svm_Tach_test_avg_acc_std'],
            'svm_Tach_test_cm_arr':shv['svm_Tach_test_cm_arr'],
            
            'svm_holdout_acc_by_label':shv['svm_holdout_acc_by_label'],
            'svm_holdout_avg_acc':shv['svm_holdout_avg_acc'],
            'svm_holdout_avg_acc_mn':shv['svm_holdout_avg_acc_mn'],
            'svm_holdout_avg_acc_std':shv['svm_holdout_avg_acc_std'],
            'svm_holdout_cm_arr':shv['svm_holdout_cm_arr'],

            'svm_test_acc_by_label':shv['svm_test_acc_by_label'],
            'svm_test_avg_acc':shv['svm_test_avg_acc'],
            'svm_test_avg_acc_mn':shv['svm_test_avg_acc_mn'],
            'svm_test_avg_acc_std':shv['svm_test_avg_acc_std'],
            'svm_test_cm_arr':shv['svm_test_cm_arr'],

            'knn_holdout_acc_by_label':shv['knn_holdout_acc_by_label'],
            'knn_holdout_avg_acc':shv['knn_holdout_avg_acc'],
            'knn_holdout_avg_acc_mn':shv['knn_holdout_avg_acc_mn'],
            'knn_holdout_avg_acc_std':shv['knn_holdout_avg_acc_std'],
            'knn_holdout_cm_arr':shv['knn_holdout_cm_arr'],

            'knn_test_acc_by_label':shv['knn_test_acc_by_label'],
            'knn_test_avg_acc':shv['knn_test_avg_acc'],
            'knn_test_avg_acc_mn':shv['knn_test_avg_acc_mn'],
            'knn_test_avg_acc_std':shv['knn_test_avg_acc_std'],
            'knn_test_cm_arr':shv['knn_test_cm_arr'],   
        }
    shv.close()
