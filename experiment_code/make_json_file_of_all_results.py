#from standard library
import sys
import json
import shelve

#from Anaconda
import numpy as np
import matplotlib.pyplot as plt

#folder with all results
sys.path.append('C:/Users/dnicho4/Documents/IPython Notebooks/test_everything/linsvm_svmrbf_knn_results_071916/')

def convert_results_dict_to_jsonifiable(dict_in):
    """
    convert_results_dict_to_jsonifiable(dict_in)
    Converts "results_dict" dictionary into something that the json module
    can parse. Does so by iterating recursively through results_dict--i.e., if it
    finds a dictionary as a value, it iterates through that dictionary as well--
    and converting any values that return "True" for "is np.ndarray" into lists
    using the numpy .tolist method. In the case of the confusion matrices, e.g.,
    "knn_test_cm_arr", it converts the array of arrays into a list of lists.
    """
    for k,v in dict_in.items():
        if type(v) is dict:
            convert_results_dict_to_jsonifiable(v)
        elif type(v) is np.ndarray:
            # check whether v is an array of arrays--i.e., the confusion matrices.
            # Note that "v.dtype is dtype('O')" would always be False
            if v.dtype==np.dtype('O') and type(v[0,0]) is np.ndarray:
                rows, cols = v.shape
                for i in range(rows):
                    for j in range(cols):
                        v[i,j] = v[i,j].tolist()
                v = v.tolist()
                dict_in[k] = v
            else: # it's just an array of scalar values
                dict_in[k] = v.tolist()
    return dict_in

#constants    
DATA_DIR = './data_for_testing/'
JSON_FNAME = './data_for_testing/data_by_bird.JSON'
RESULTS_DIR = './linsvm_svmrbf_knn_results_071916/'
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
            'labelset':shv['labelset'],
            'non_intro_note_labelset':shv['non_intro_note_labelset'],
            
#            'svm_Tach_holdout_rnd_acc':shv['svm_Tach_holdout_rnd_acc'],
#            'svm_Tach_holdout_rnd_acc_mn':shv['svm_Tach_holdout_rnd_acc_mn'],
#            'svm_Tach_holdout_rnd_acc_std':shv['svm_Tach_holdout_rnd_acc_std'],            
#            
#            'svm_Tach_test_rnd_acc':shv['svm_Tach_test_rnd_acc'],
#            'svm_Tach_test_rnd_acc_mn':shv['svm_Tach_test_rnd_acc_mn'],
#            'svm_Tach_test_rnd_acc_std':shv['svm_Tach_test_rnd_acc_std'],
#            
#            'svm_holdout_rnd_acc':shv['svm_holdout_rnd_acc'],
#            'svm_holdout_rnd_acc_mn':shv['svm_holdout_rnd_acc_mn'],
#            'svm_holdout_rnd_acc_std':shv['svm_holdout_rnd_acc_std'],

#            'svm_test_rnd_acc':shv['svm_test_rnd_acc'],
#            'svm_test_rnd_acc_mn':shv['svm_test_rnd_acc_mn'],
#            'svm_test_rnd_acc_std':shv['svm_test_rnd_acc_std'],

#            'knn_holdout_rnd_acc':shv['knn_holdout_rnd_acc'],
#            'knn_holdout_rnd_acc_mn':shv['knn_holdout_rnd_acc_mn'],
#            'knn_holdout_rnd_acc_std':shv['knn_holdout_rnd_acc_std'],
#            
#            'knn_test_rnd_acc':shv['knn_test_rnd_acc'],
#            'knn_test_rnd_acc_mn':shv['knn_test_rnd_acc_mn'],
#            'knn_test_rnd_acc_std':shv['knn_test_rnd_acc_std'],

#            'linsvm_holdout_acc_by_label':shv['linsvm_holdout_acc_by_label'],
#            'linsvm_holdout_avg_acc':shv['linsvm_holdout_avg_acc'],
#            'linsvm_holdout_avg_acc_mn':shv['linsvm_holdout_avg_acc_mn'],
#            'linsvm_holdout_avg_acc_std':shv['linsvm_holdout_avg_acc_std'],
#            'linsvm_holdout_cm_arr':shv['linsvm_holdout_cm_arr'],

            'linsvm_test_rnd_acc':shv['linsvm_test_rnd_acc'],
            'linsvm_test_rnd_acc_mn':shv['linsvm_test_rnd_acc_mn'],
            'linsvm_test_rnd_acc_std':shv['linsvm_test_rnd_acc_std'],

            'linsvm_test_no_intro_rnd_acc':shv['linsvm_test_no_intro_rnd_acc'],
            'linsvm_test_no_intro_rnd_acc_mn':shv['linsvm_test_no_intro_rnd_acc_mn'],
            'linsvm_test_no_intro_rnd_acc_std':shv['linsvm_test_no_intro_rnd_acc_std'],

            'linsvm_train_acc_by_label':shv['linsvm_train_acc_by_label'],
            'linsvm_train_avg_acc':shv['linsvm_train_avg_acc'],
            'linsvm_train_avg_acc_mn':shv['linsvm_train_avg_acc_mn'],
            'linsvm_train_avg_acc_std':shv['linsvm_train_avg_acc_std'],

            'linsvm_test_acc_by_label':shv['linsvm_test_acc_by_label'],
            'linsvm_test_avg_acc':shv['linsvm_test_avg_acc'],
            'linsvm_test_avg_acc_mn':shv['linsvm_test_avg_acc_mn'],
            'linsvm_test_avg_acc_std':shv['linsvm_test_avg_acc_std'],
#            'linsvm_test_cm_arr':shv['linsvm_test_cm_arr'],

#            'linsvm_holdout_no_intro_acc_by_label':shv['linsvm_holdout_no_intro_acc_by_label'],
#            'linsvm_holdout_no_intro_avg_acc':shv['linsvm_holdout_no_intro_avg_acc'],
#            'linsvm_holdout_no_intro_avg_acc_mn':shv['linsvm_holdout_no_intro_avg_acc_mn'],
#            'linsvm_holdout_no_intro_avg_acc_std':shv['linsvm_holdout_no_intro_avg_acc_std'],
#            'linsvm_holdout_no_intro_cm_arr':shv['linsvm_holdout_no_intro_cm_arr'],

            'linsvm_test_rnd_acc_flat':shv['linsvm_test_rnd_acc_flat'],
            'num_train_samples_flat':shv['num_train_samples_flat'],
            'linsvm_train_sample_bins':shv['linsvm_train_sample_bins'],
            'linsvm_test_rnd_acc_by_sample_mn':shv['linsvm_test_rnd_acc_by_sample_mn'],
            'linsvm_test_rnd_acc_by_sample_std':shv['linsvm_test_rnd_acc_by_sample_std'],

            'linsvm_test_no_intro_acc_by_label':shv['linsvm_test_no_intro_acc_by_label'],
            'linsvm_test_no_intro_avg_acc':shv['linsvm_test_no_intro_avg_acc'],
            'linsvm_test_no_intro_avg_acc_mn':shv['linsvm_test_no_intro_avg_acc_mn'],
            'linsvm_test_no_intro_avg_acc_std':shv['linsvm_test_no_intro_avg_acc_std'],
            'linsvm_test_no_intro_cm_arr':shv['linsvm_test_no_intro_cm_arr'],

            'svm_Tach_train_acc_by_label':shv['svm_Tach_train_acc_by_label'],
            'svm_Tach_train_avg_acc':shv['svm_Tach_train_avg_acc'],
            'svm_Tach_train_avg_acc_mn':shv['svm_Tach_train_avg_acc_mn'],
            'svm_Tach_train_avg_acc_std':shv['svm_Tach_train_avg_acc_std'],
            'svm_Tach_train_cm_arr':shv['svm_Tach_train_cm_arr'],
            
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

            'knn_train_acc_by_label':shv['knn_train_acc_by_label'],
            'knn_train_avg_acc':shv['knn_train_avg_acc'],
            'knn_train_avg_acc_mn':shv['knn_train_avg_acc_mn'],
            'knn_train_avg_acc_std':shv['knn_train_avg_acc_std'],
            'knn_train_cm_arr':shv['knn_train_cm_arr'], 

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

results_dict_for_jsonify = convert_results_dict_to_jsonifiable(results_dict)
with open('results_dict.json','w') as outfile:
    json.dump(results_dict_for_jsonify,outfile)
