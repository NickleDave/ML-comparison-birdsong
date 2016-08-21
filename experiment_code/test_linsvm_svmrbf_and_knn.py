#from standard library
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
RESULTS_DIR = './linsvm_svmrbf_knn_results/'
JSON_FNAME = './data_for_testing/data_by_bird.JSON'
RESULTS_SHELVE_BASE_FNAME = 'linsvm_svmrbf_knn_results_'
TRAIN_PARAMS = parameter('-s 1 -c 1 -q') # for liblinear library function
DURATION_COLUMN_INDEX = 512 # in Tachibana feature set, column 512 ends up being the one that holds duration

### load JSON file that contains filenames of training/testing data
### and names of labels for samples, i.e., birdsong syllable names
with open(JSON_FNAME) as json_file:
    data_by_bird = json.load(json_file)

# constants used in main loop
NUM_SONGS_TO_TEST = list(range(3,16,3)) + [21,27,33,39]
# i.e., [3,6,9,...15,21,27,33,39].
REPLICATES = range(1,6)
HOLDOUT_TEST_SIZE = 0.4 # 40% of training data set used as holdout test set

# scalers from scikit used in main loop
svm_scaler = StandardScaler()
svm_Tach_scaler = StandardScaler()
knn_scaler = StandardScaler()


### main loop
for birdID, bird_data in data_by_bird.items():
    print("analyzing: " + birdID)

    #load train/test data, label names
    svm_train_fname = os.path.join(DATA_DIR + bird_data['svm_train_feat'])
    svm_test_fname = os.path.join(DATA_DIR + bird_data['svm_test_feat'])
    knn_train_fname = os.path.join(DATA_DIR + bird_data['knn_train_feat'])
    knn_test_fname = os.path.join(DATA_DIR + bird_data['knn_test_feat'])
    labelset = list(bird_data['labelset'])
    labelset = [ord(label) for label in labelset]
    intro_labels = list(bird_data['intro labels'])
    intro_labels = [ord(label) for label in intro_labels] 

    svm_train_samples,svm_train_labels,svm_train_song_IDs = load_from_mat(svm_train_fname)
    svm_train_samples,svm_train_labels,svm_train_song_IDs = filter_samples(svm_train_samples,
                                                                            svm_train_labels,
                                                                            labelset,svm_train_song_IDs)
    svm_test_samples,svm_test_labels,svm_test_song_IDs = load_from_mat(svm_test_fname)
    svm_test_samples,svm_test_labels = filter_samples(svm_test_samples,
                                                        svm_test_labels,
                                                        labelset,
                                                        svm_test_song_IDs)[0:2] # don't need song_IDs for test set
    svm_test_samples_Tach = svm_test_samples[:,0:532] # just Tachibana features
    linsvm_test_labels = svm_test_labels.tolist() # liblinear library functions take a list of labels, not an array
    
    knn_train_samples, knn_train_labels, knn_train_song_IDs = load_knn_data(knn_train_fname,labelset)
    knn_test_samples, knn_test_labels = load_knn_data(knn_test_fname,labelset)[0:2] # don't need song_IDs

    assert svm_train_song_IDs.shape == knn_train_song_IDs.shape # should be the same since samples are taken from exact same song files

    for ind, num_songs in enumerate(NUM_SONGS_TO_TEST):
        print("Testing accuracy for training set composed of " + str(num_songs) + " songs")
        for replicate in REPLICATES:
            print("Replicate " + str(replicate) + ". ")
            # below in call to train_test_song_split, note that "num_songs" is # used to train and 
            # HOLDOUT_TEST_SIZE is # used to test (from original training set)
            svm_train_samples_subset,svm_train_labels_subset,svm_holdout_samples,svm_holdout_labels,train_sample_IDs,holdout_sample_IDs = \
                train_test_song_split(svm_train_samples,svm_train_labels,svm_train_song_IDs,num_songs,HOLDOUT_TEST_SIZE)
            svm_train_samples_subset_scaled = svm_scaler.fit_transform(svm_train_samples_subset)
            svm_train_samples_subset_Tach = svm_train_samples_subset_scaled[:,0:532] # just Tachibana features
            svm_holdout_samples_scaled = svm_scaler.transform(svm_holdout_samples)
            svm_holdout_samples_Tach = svm_holdout_samples_scaled[:,0:532]

            ### test support vector machine with linear kernel using liblinear ###
            ### i.e., what Tachibana et al. 2014 did ###
            # need to convert train samples to 'dictlist',
            # a list of dictionaries where each dictionary represents a training sample,
            # for liblinear library functions
            linsvm_train_samples = array_to_dictlist(svm_train_samples_subset_Tach)
            linsvm_holdout_samples = array_to_dictlist(svm_holdout_samples_Tach)
            linsvm_test_samples = array_to_dictlist(svm_test_samples_Tach)
            #convert labels to list to use with liblinear train function
            linsvm_train_labels = svm_train_labels_subset.tolist()
            linsvm_holdout_labels = svm_holdout_labels.tolist()
            prob = problem(linsvm_train_labels,linsvm_train_samples)
            print(" Training linear SVM model using " + str(len(linsvm_train_samples)) + " samples.\n")
            model = train(prob,TRAIN_PARAMS)
            print(" Testing predictions on training set: ")
            linsvm_train_pred_labels,linsvm_train_acc,linsvm_train_vals = \
                predict(linsvm_train_labels,linsvm_train_samples,model)
            print(" Testing predictions on holdout set: ")
            linsvm_holdout_pred_labels,linsvm_holdout_acc,linsvm_holdout_vals = \
                predict(linsvm_holdout_labels,linsvm_holdout_samples,model)
            print(" Testing predictions on larger test set: ")
            linsvm_test_pred_labels,linsvm_test_acc,linsvm_test_vals = \
                predict(linsvm_test_labels,linsvm_test_samples,model)

            ### test support vector machine with linear kernel using liblinear ###
            ### without intro notes!
            linsvm_train_samples_no_intro,linsvm_train_labels_no_intro = \
                filter_samples(svm_train_samples_subset_Tach,svm_train_labels_subset,intro_labels,remove=True)
            linsvm_train_samples_no_intro = array_to_dictlist(linsvm_train_samples_no_intro)
            # get duration of samples without intro notes in case I want to see acc v. that duration
            train_sample_no_intro_total_duration = sum(svm_train_samples[train_sample_IDs,DURATION_COLUMN_INDEX])
            linsvm_holdout_samples_no_intro,linsvm_holdout_labels_no_intro = \
                filter_samples(svm_holdout_samples_Tach,svm_holdout_labels,intro_labels,remove=True)
            linsvm_holdout_samples_no_intro =  array_to_dictlist(linsvm_holdout_samples_no_intro)
            linsvm_test_samples_no_intro,linsvm_test_labels_no_intro = \
                filter_samples(svm_test_samples_Tach,svm_test_labels,intro_labels,remove=True)
            linsvm_test_samples_no_intro = array_to_dictlist(linsvm_test_samples_no_intro)
            #convert labels to list to use with liblinear train function
            linsvm_train_labels_no_intro = linsvm_train_labels_no_intro.tolist()
            linsvm_holdout_labels_no_intro = linsvm_holdout_labels_no_intro.tolist()
            linsvm_test_labels_no_intro = linsvm_test_labels_no_intro.tolist()
            prob = problem(linsvm_train_labels_no_intro,linsvm_train_samples_no_intro)
            print(" Training linear SVM model using " + str(len(linsvm_train_samples)) + " samples, intro. notes removed.")
            model = train(prob,TRAIN_PARAMS)
            print(" Testing predictions on training set with intro. notes removed: ")
            linsvm_train_no_intro_pred_labels,linsvm_train_no_intro_acc,linsvm_train_no_intro_vals = \
                predict(linsvm_train_labels_no_intro,linsvm_train_samples_no_intro,model)
            print(" Testing predictions on holdout set with intro. notes removed: ")
            linsvm_holdout_no_intro_pred_labels,linsvm_holdout_no_intro_acc,linsvm_holdout_no_intro_vals = \
                predict(linsvm_holdout_labels_no_intro,linsvm_holdout_samples_no_intro,model)
            print(" Testing predictions on larger test set with intro. notes removed: ")
            linsvm_test_no_intro_pred_labels,linsvm_test_no_intro_acc,linsvm_test_no_intro_vals = \
                predict(linsvm_test_labels_no_intro,linsvm_test_samples_no_intro,model)

            ### test support vector machine with radial basis function ###
            ### using just Tachibana features ###
            print(" Training SVM w/RBF using just Tachibana features.")           
            best_params_Tach, best_grid_score_Tach = grid_search(svm_train_samples_subset_Tach,svm_train_labels_subset)
            svm_Tach_clf = SVC(C=best_params_Tach['C'],gamma=best_params_Tach['gamma'],decision_function_shape='ovr')
            svm_Tach_clf.fit(svm_train_samples_subset_Tach,svm_train_labels_subset)
            svm_Tach_train_pred_labels = svm_Tach_clf.predict(svm_train_samples_subset_Tach)
            svm_Tach_train_score = svm_Tach_clf.score(svm_train_samples_subset_Tach,svm_train_labels_subset)
            svm_Tach_holdout_pred_labels = svm_Tach_clf.predict(svm_holdout_samples_Tach)
            svm_Tach_holdout_score = svm_Tach_clf.score(svm_holdout_samples_Tach,svm_holdout_labels)
            svm_Tach_test_pred_labels = svm_Tach_clf.predict(svm_test_samples_Tach)
            svm_Tach_test_score = svm_Tach_clf.score(svm_test_samples_Tach,svm_test_labels)
            svm_Tach_decision_func = svm_Tach_clf.decision_function(svm_test_samples_Tach)
            print(" svm score on train set: ",svm_Tach_train_score)
            print(" svm score on holdout set: ",svm_Tach_holdout_score)
            print(" svm score on test set: ",svm_Tach_test_score)

            ### test support vector machine with radial basis function ###
            ### using Tachibana features plus adjacent syllable features ###           
            print(" Training SVM w/RBF using Tachibana features plus features of adjacent syllables.")
            best_params, best_grid_score = grid_search(svm_train_samples_subset_scaled,svm_train_labels_subset)
            svm_clf = SVC(C=best_params['C'],gamma=best_params['gamma'],decision_function_shape='ovr')
            svm_clf.fit(svm_train_samples_subset_scaled,svm_train_labels_subset)
            svm_holdout_pred_labels = svm_clf.predict(svm_holdout_samples_scaled)
            svm_holdout_score = svm_clf.score(svm_holdout_samples_scaled,svm_holdout_labels)
            svm_test_pred_labels = svm_clf.predict(svm_test_samples_scaled)
            svm_test_score = svm_clf.score(svm_test_samples_scaled,svm_test_labels)
            svm_decision_func = svm_clf.decision_function(svm_test_samples_scaled)
            print(" svm score on holdout set: ",svm_holdout_score)
            print(" svm score on test set: ",svm_test_score)

            ### test k-Nearest neighbors ###
            ### using common songbird acoustic analysis parameters as features ###
            ### including some of those parameters from adjacent syllables in each sample ###
            print(" 'Training' k-NN model using acoustic params + adjacent syllable features.")
            knn_train_samples_subset = knn_train_samples[train_sample_IDs,:]
            knn_train_labels_subset = knn_train_labels[train_sample_IDs]
            knn_train_samples_scaled = knn_scaler.fit_transform(knn_train_samples_subset)       
            knn_holdout_samples = knn_train_samples[holdout_sample_IDs,:]
            knn_holdout_labels = knn_train_labels[holdout_sample_IDs]
            knn_holdout_samples_scaled = knn_scaler.transform(knn_holdout_samples)
            knn_test_samples_scaled = knn_scaler.transform(knn_test_samples)
            print(" finding best k")
            k = find_best_k(knn_train_samples_scaled,knn_train_labels_subset,knn_holdout_samples,knn_holdout_labels)[1]
            # ^ [1] because I don't want mn_scores, just k
            print(" best k was: " + str(k))
            knn_clf = neighbors.KNeighborsClassifier(k,'distance')
            knn_clf.fit(knn_train_samples_scaled,knn_train_labels_subset)    
            knn_train_pred_labels = knn_clf.predict(knn_train_samples_scaled)
            knn_train_score = knn_clf.score(knn_train_samples_scaled,knn_train_labels_subset)
            knn_holdout_pred_labels = knn_clf.predict(knn_holdout_samples_scaled)
            knn_holdout_score = knn_clf.score(knn_holdout_samples_scaled,knn_holdout_labels)
            knn_test_pred_labels = knn_clf.predict(knn_test_samples_scaled)
            knn_test_score = knn_clf.score(knn_test_samples_scaled,knn_test_labels)
            print(" knn score on train set: ",knn_train_score)
            print(" knn score on holdout set: ",knn_holdout_score)
            print(" knn score on test set: ",knn_test_score)

            ### save results from this replicate ###
            results_shelve_fname = \
                RESULTS_DIR + RESULTS_SHELVE_BASE_FNAME + str(birdID) + ", " + str(num_songs) + ' songs, replicate ' + str(replicate) + '.db'                        
            with shelve.open(results_shelve_fname) as shlv:
                shlv['train_sample_IDs'] = train_sample_IDs
                shlv['holdout_sample_IDs'] = holdout_sample_IDs
                shlv['train_sample_total_duration'] = train_sample_total_duration
                shlv['train_sample_no_intro_total_duration'] = train_sample_no_intro_total_duration 

                shlv['linsvm_train_pred_labels'] = linsvm_train_pred_labels
                shlv['linsvm_train_acc'] = linsvm_train_acc
                shlv['linsvm_train_vals'] = linsvm_train_vals
                shlv['linsvm_holdout_pred_labels'] = linsvm_holdout_pred_labels
                shlv['linsvm_holdout_acc'] = linsvm_holdout_acc
                shlv['linsvm_holdout_vals'] = linsvm_holdout_vals
                shlv['linsvm_test_pred_labels'] = linsvm_test_pred_labels
                shlv['linsvm_test_acc'] = linsvm_test_acc
                shlv['linsvm_test_vals'] = linsvm_test_vals

                shlv['linsvm_train_no_intro_pred_labels'] = linsvm_train_no_intro_pred_labels
                shlv['linsvm_train_no_intro_acc'] = linsvm_train_no_intro_acc
                shlv['linsvm_train_no_intro_vals'] = linsvm_train_no_intro_vals
                shlv['linsvm_holdout_no_intro_pred_labels'] = linsvm_holdout_no_intro_pred_labels
                shlv['linsvm_holdout_no_intro_acc'] = linsvm_holdout_no_intro_acc
                shlv['linsvm_holdout_no_intro_vals'] = linsvm_holdout_no_intro_vals
                shlv['linsvm_test_no_intro_pred_labels'] = linsvm_test_no_intro_pred_labels
                shlv['linsvm_test_no_intro_acc'] = linsvm_test_no_intro_acc
                shlv['linsvm_test_no_intro_vals'] = linsvm_test_no_intro_vals

                shlv['best_params'] = best_params
                shlv['svm_holdout_pred_labels'] = svm_holdout_pred_labels
                shlv['svm_holdout_score'] = svm_holdout_score
                shlv['svm_test_pred_labels'] = svm_test_pred_labels
                shlv['svm_test_score'] = svm_test_score
                shlv['svm_decision_func'] = svm_decision_func
                
                shlv['best_params_Tach'] = best_params_Tach
                shlv['svm_Tach_train_pred_labels'] = svm_Tach_train_pred_labels
                shlv['svm_Tach_train_score'] = svm_Tach_train_score
                shlv['svm_Tach_holdout_pred_labels'] = svm_Tach_holdout_pred_labels
                shlv['svm_Tach_holdout_score'] = svm_Tach_holdout_score
                shlv['svm_Tach_test_pred_labels'] = svm_Tach_test_pred_labels
                shlv['svm_Tach_test_score'] = svm_Tach_test_score
                shlv['svm_Tach_decision_func'] = svm_Tach_decision_func

                shlv['knn_train_pred_labels'] = knn_train_pred_labels
                shlv['knn_train_score'] = knn_train_score
                shlv['knn_holdout_pred_labels'] = knn_holdout_pred_labels
                shlv['knn_holdout_score'] = knn_holdout_score
                shlv['knn_test_pred_labels'] = knn_test_pred_labels
                shlv['knn_test_score'] = knn_test_score
