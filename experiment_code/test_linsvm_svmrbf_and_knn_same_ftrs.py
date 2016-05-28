#from standard library
import pdb
import sys
import json
import shelve
import os

#from Anaconda
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import neighbors

#from liblinear
sys.path.append("C:/Program Files/liblinear-2.01/python")
from liblinearutil import *

#from functions written just for these experiments
from liblinear_test_utility_functions import filter_samples,scale_data,array_to_dictlist,remove_samples_by_label,dictlist_to_array
from svm_rbf_test_utility_functions import load_from_mat,filter_samples,grid_search,train_test_song_split
from knn_test_functions import load_knn_data, find_best_k

#constants
DATA_DIR = './data_for_testing/'
TARGET_RESULTS_DIR = './linsvm_svmrbf_knn_same_ftr_results/'
JSON_FNAME = './data_for_testing/data_by_bird.JSON'
RESULTS_SHELVE_BASE_FNAME = 'test_same_ftr_results_'
TRAIN_PARAMS = parameter('-s 1 -c 1 -q') # for liblinear library function
DURATION_COLUMN_INDEX = 0 # after removing other Tachibana features (spectrograms), first index (0) is duration

### load JSON file that contains filenames of training/testing data
### and names of labels for samples, i.e., birdsong syllable names
with open(JSON_FNAME) as json_file:
    data_by_bird = json.load(json_file)

# constants used in main loop
NUM_SONGS_TO_TEST = list(range(3,16,3)) + [21,27,33,39]
# i.e., [3,6,9,...15,21,27,33,39].
REPLICATES = range(0,10)
HOLDOUT_TEST_SIZE = 0.4 # 40% of training data set used as holdout test set

# scalers from scikit used in main loop
scaler = StandardScaler()
scaler_no_intro = StandardScaler()

for birdID, bird_data in data_by_bird.items():
    print("analyzing: " + birdID)

    train_fname = os.path.join(DATA_DIR + bird_data['svm_train_feat'])
    test_fname = os.path.join(DATA_DIR + bird_data['svm_test_feat'])
    labelset = list(bird_data['labelset'])
    labelset = [ord(label) for label in labelset]
    intro_labels = list(bird_data['intro labels'])
    intro_labels = [ord(label) for label in intro_labels] 
    
    train_samples,train_labels,train_song_IDs = load_from_mat(train_fname)
    train_samples,train_labels,train_song_IDs = filter_samples(train_samples,train_labels,labelset,train_song_IDs)
    train_samples = train_samples[:,-24:-4] # just acoustic features from Tachibana + duration and no. of zero crossings
    test_samples,test_labels,test_song_IDs = load_from_mat(test_fname)
    test_samples,test_labels = filter_samples(test_samples,test_labels,labelset,test_song_IDs)[0:2] # don't need song_IDs for test set
    linsvm_test_labels = test_labels.tolist() # liblinear library functions take a list of labels, not an array
    test_samples = test_samples[:,-24:-4] # just acoustic features from Tachibana + duration and no. of zero crossings


    for ind, num_songs in enumerate(NUM_SONGS_TO_TEST):
        print("Testing accuracy for training set composed of " + str(num_songs) + " songs")
        for replicate in REPLICATES:
            print("Replicate " + str(replicate + 1) + ". ")
            # below in call to train_test_song_split, note that "num_songs" is # used to train and 
            # HOLDOUT_TEST_SIZE is # used to test (from original training set)
            train_samples_subset,train_labels_subset,holdout_samples,holdout_labels,train_sample_IDs,holdout_sample_IDs = \
                train_test_song_split(train_samples,train_labels,train_song_IDs,num_songs,HOLDOUT_TEST_SIZE)
            train_samples_subset_scaled = scaler.fit_transform(train_samples_subset)
            holdout_samples_scaled = scaler.transform(holdout_samples)
            test_samples_scaled = scaler.transform(test_samples)
           # get duration of samples in case I want to see acc v. that duration
            train_sample_total_duration = sum(train_samples[train_sample_IDs,DURATION_COLUMN_INDEX])

            ### test support vector machine with linear kernel using liblinear ###
            ### i.e., what Tachibana et al. 2014 did ###
            # need to convert train samples to 'dictlist',
            # a list of dictionaries where each dictionary represents a training sample,
            # for liblinear library functions
            linsvm_train_samples = array_to_dictlist(train_samples_subset_scaled)
            linsvm_holdout_samples = array_to_dictlist(holdout_samples_scaled)
            linsvm_test_samples = array_to_dictlist(test_samples_scaled)
            #convert labels to list to use with liblinear train function
            linsvm_train_labels = train_labels_subset.tolist()
            linsvm_holdout_labels = holdout_labels.tolist()
            prob = problem(linsvm_train_labels,linsvm_train_samples)
            print(" Training linear SVM model using " + str(len(linsvm_train_samples)) + " samples.\n")
            model = train(prob,TRAIN_PARAMS)
            print(" Testing predictions on holdout set: ")
            linsvm_holdout_pred_labels,linsvm_holdout_acc,linsvm_holdout_vals = \
                predict(linsvm_holdout_labels,linsvm_holdout_samples,model)
            print(" Testing predictions on larger test set: ")
            linsvm_test_pred_labels,linsvm_test_acc,linsvm_test_vals = \
                predict(linsvm_test_labels,linsvm_test_samples,model)

            ### test support vector machine with linear kernel using liblinear ###
            ### i.e., what Tachibana et al. 2014 did ###
            # need to convert train samples to 'dictlist',
            # a list of dictionaries where each dictionary represents a training sample,
            # for liblinear library functions
            linsvm_train_samples_no_intro,linsvm_train_labels_no_intro = \
                filter_samples(train_samples_subset,train_labels_subset,intro_labels,remove=True)
            linsvm_train_samples_no_intro = scaler_no_intro.fit_transform(linsvm_train_samples_no_intro)
            linsvm_train_samples_no_intro = array_to_dictlist(linsvm_train_samples_no_intro)
            # get duration of samples without intro notes in case I want to see acc v. that duration
            train_sample_no_intro_total_duration = sum(train_samples[train_sample_IDs,DURATION_COLUMN_INDEX])
            linsvm_holdout_samples_no_intro,linsvm_holdout_labels_no_intro = \
                filter_samples(holdout_samples,holdout_labels,intro_labels,remove=True)
            linsvm_holdout_samples_no_intro = scaler_no_intro.transform(linsvm_holdout_samples_no_intro)
            linsvm_holdout_samples_no_intro =  array_to_dictlist(linsvm_holdout_samples_no_intro)
            linsvm_test_samples_no_intro,linsvm_test_labels_no_intro = \
                filter_samples(test_samples,test_labels,intro_labels,remove=True)
            linsvm_test_samples_no_intro = scaler_no_intro.transform(linsvm_test_samples_no_intro)
            linsvm_test_samples_no_intro = array_to_dictlist(linsvm_test_samples_no_intro)
            #convert labels to list to use with liblinear train function
            linsvm_train_labels_no_intro = linsvm_train_labels_no_intro.tolist()
            linsvm_holdout_labels_no_intro = linsvm_holdout_labels_no_intro.tolist()
            linsvm_test_labels_no_intro = linsvm_test_labels_no_intro.tolist()
            prob = problem(linsvm_train_labels_no_intro,linsvm_train_samples_no_intro)
            print(" Training linear SVM model using " + str(len(linsvm_train_samples)) + " samples, intro. notes removed.")
            model = train(prob,TRAIN_PARAMS)
            print(" Testing predictions on holdout set with intro. notes removed: ")
            linsvm_holdout_no_intro_pred_labels,linsvm_holdout_no_intro_acc,linsvm_holdout_no_intro_vals = \
                predict(linsvm_holdout_labels_no_intro,linsvm_holdout_samples_no_intro,model)
            print(" Testing predictions on larger test set with intro. notes removed: ")
            linsvm_test_no_intro_pred_labels,linsvm_test_no_intro_acc,linsvm_test_no_intro_vals = \
                predict(linsvm_test_labels_no_intro,linsvm_test_samples_no_intro,model)
            
            ### test k-Nearest neighbors ###
            print(" 'Training' k-NN model using acoustic params + adjacent syllable features.")
            print(" finding best k")
            k = find_best_k(train_samples_subset_scaled,train_labels_subset,holdout_samples_scaled,holdout_labels)[1]
            # ^ [1] because I don't want mn_scores, just k
            print(" best k was: " + str(k))
            knn_clf = neighbors.KNeighborsClassifier(k,'distance')
            knn_clf.fit(train_samples_subset_scaled,train_labels_subset)    
            knn_holdout_pred_labels = knn_clf.predict(holdout_samples_scaled)
            knn_holdout_score = knn_clf.score(holdout_samples_scaled,holdout_labels)
            knn_test_pred_labels = knn_clf.predict(test_samples_scaled)
            knn_test_score = knn_clf.score(test_samples_scaled,test_labels)
            print(" knn score on holdout set: ",knn_holdout_score)
            print(" knn score on test set: ",knn_test_score)

            ### test support vector machine with radial basis function ###
            print("executing grid search for SVM")
            best_params, best_grid_score = grid_search(train_samples_subset_scaled,train_labels_subset)
            svm_clf = SVC(C=best_params['C'],gamma=best_params['gamma'],decision_function_shape='ovr')
            svm_clf.fit(train_samples_subset_scaled,train_labels_subset)
            svm_holdout_pred_labels = svm_clf.predict(holdout_samples_scaled)
            svm_holdout_score = svm_clf.score(holdout_samples_scaled,holdout_labels)
            svm_test_pred_labels = svm_clf.predict(test_samples_scaled)
            svm_test_score = svm_clf.score(test_samples_scaled,test_labels)
            svm_decision_func = svm_clf.decision_function(test_samples_scaled)
            print("svm score on holdout set: ",svm_holdout_score)
            print("svm score on test set: ",svm_test_score)
          
            ### save results from this replicate ###
            results_shelve_fname = \
                TARGET_RESULTS_DIR + RESULTS_SHELVE_BASE_FNAME + str(birdID) + ", " + str(num_songs) + ' songs, replicate ' + str(replicate + 1) + '.db'                        
            with shelve.open(results_shelve_fname) as shlv:
                shlv['train_sample_IDs'] = train_sample_IDs
                shlv['holdout_sample_IDs'] = holdout_sample_IDs
                shlv['train_sample_total_duration'] = train_sample_total_duration
                shlv['train_sample_no_intro_total_duration'] = train_sample_no_intro_total_duration 

                shlv['linsvm_holdout_pred_labels'] = linsvm_holdout_pred_labels
                shlv['linsvm_holdout_acc'] = linsvm_holdout_acc
                shlv['linsvm_holdout_vals'] = linsvm_holdout_vals
                shlv['linsvm_test_pred_labels'] = linsvm_test_pred_labels
                shlv['linsvm_test_acc'] = linsvm_test_acc
                shlv['linsvm_test_vals'] = linsvm_test_vals

                shlv['linsvm_holdout_no_intro_pred_labels'] = linsvm_holdout_no_intro_pred_labels
                shlv['linsvm_holdout_no_intro_acc'] = linsvm_holdout_no_intro_acc
                shlv['linsvm_holdout_no_intro_vals'] = linsvm_holdout_no_intro_vals
                shlv['linsvm_test_no_intro_pred_labels'] = linsvm_test_no_intro_pred_labels
                shlv['linsvm_test_no_intro_acc'] = linsvm_test_no_intro_acc
                shlv['linsvm_test_no_intro_vals'] = linsvm_test_no_intro_vals

                shlv['k'] = k
                shlv['knn_holdout_pred_labels'] = knn_holdout_pred_labels
                shlv['knn_holdout_score'] = knn_holdout_score
                shlv['knn_test_pred_labels'] = knn_test_pred_labels
                shlv['knn_test_score'] = knn_test_score

                shlv['best_params'] = best_params
                shlv['svm_holdout_pred_labels'] = svm_holdout_pred_labels
                shlv['svm_holdout_score'] = svm_holdout_score
                shlv['svm_test_pred_labels'] = svm_test_pred_labels
                shlv['svm_test_score'] = svm_test_score
                shlv['svm_decision_func'] = svm_decision_func          

