#from standard library
import pdb
import os
import json
import shelve

# from Anaconda distrib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as confuse_mat
from sklearn.metrics import recall_score

# from functions files written for these experiments
from svm_rbf_test_utility_functions import load_from_mat
from knn_test_functions import load_knn_data

####utility functions for this script
def filter_labels(labels,labelset):
    """
    filter_labels(labels,labelset)
    returns labels with any elements removed that are not in labelset
    """
    labels_to_keep = np.in1d(labels,labelset) #returns boolean vector, True where label is in labelset
    labels = labels[labels_to_keep]
    return labels
    
def get_acc_by_label(labels,pred_labels,labelset):
    """
    get_acc_by_label(labels,pred_labels,labelset)

    arguments:
    labels -- vector of labels from a test data set
    pred_labels -- vector of predicted labels returned by algorithm given samples from test data set
    labelset -- set of unique labels from test data set, i.e., numpy.unique(labels)

    returns:
    acc_by_label -- vector of accuracies
    avg_acc -- average accuracy across labels, i.e., numpy.mean(acc_by_label)
    """
    acc_by_label = np.zeros((len(labelset)))
    for ind,label in enumerate(labelset):
        label_ids = np.in1d(labels,label) #find all occurences of label in test data
        pred_for_that_label = pred_labels[label_ids]
        matches = pred_for_that_label==label
        #sum(matches) is equal to number of true positives
        #len(matches) is equal to number of true positives and false negatives
        acc = sum(matches) / len(matches)
        acc_by_label[ind] = acc
    avg_acc = np.mean(acc_by_label)
    return acc_by_label,avg_acc

#constants
DATA_DIR = './data_for_testing/'
RESULTS_DIR = './linsvm_svmrbf_knn_with_knn_ftr_results/'
JSON_FNAME = './data_for_testing/data_by_bird.JSON'
SHELVE_BASE_FNAME = 'test_knn_ftr_results_'
RESULTS_SHV_BASE_FNAME = '_knn_ftr_summary_results'

### load JSON file that contains filenames of training/testing data
### and names of labels for samples, i.e., birdsong syllable names
with open(JSON_FNAME) as json_file:
    data_by_bird = json.load(json_file)

# constants used in main loop
NUM_SONGS_TO_TEST = list(range(3,16,3)) + [21,27,33,39]
# i.e., [3,6,9,...15,21,27,33,39].
REPLICATES = range(1,11)

# to initialize arrays
rows = len(REPLICATES)
cols = len(NUM_SONGS_TO_TEST)

### main loop
for bird_ID, bird_data in data_by_bird.items():
    svm_train_fname = os.path.join(DATA_DIR + bird_data['svm_train_feat'])
    svm_test_fname = os.path.join(DATA_DIR + bird_data['svm_test_feat'])
    knn_train_fname = os.path.join(DATA_DIR + bird_data['knn_train_feat'])
    knn_test_fname = os.path.join(DATA_DIR + bird_data['knn_test_feat'])
    labelset = list(bird_data['labelset'])
    labelset = [ord(label) for label in labelset]  
    non_intro_note_labelset = list(bird_data['not intro labels']) # for use with data from linsvm_no_intro tests
    non_intro_note_labelset = [ord(label) for label in non_intro_note_labelset]
    
    #pre-allocate arrays to hold values
    #each array is a matrix of m replicates * n conditions, i.e., number of songs used to train model
    num_train_samples = np.zeros((rows,cols)) # rows = replicates, because boxplot plots by column
    train_sample_total_duration = np.zeros((rows,cols))
    train_sample_no_intro_total_duration = np.zeros((rows,cols))

    #Rand accuracy
    linsvm_holdout_rnd_acc = np.zeros((rows,cols))
    linsvm_test_rnd_acc = np.zeros((rows,cols))
    linsvm_holdout_no_intro_rnd_acc = np.zeros((rows,cols))
    linsvm_test_no_intro_rnd_acc = np.zeros((rows,cols))
    svm_holdout_rnd_acc = np.zeros((rows,cols))
    svm_test_rnd_acc = np.zeros((rows,cols))
    knn_holdout_rnd_acc = np.zeros((rows,cols))
    knn_test_rnd_acc = np.zeros((rows,cols))

    #average of per-label accuracy ( true positive / (true positive + false negative))
    linsvm_holdout_avg_acc = np.zeros((rows,cols))
    linsvm_test_avg_acc = np.zeros((rows,cols))
    linsvm_holdout_no_intro_avg_acc = np.zeros((rows,cols))
    linsvm_test_no_intro_avg_acc = np.zeros((rows,cols))
    svm_holdout_avg_acc = np.zeros((rows,cols))
    svm_test_avg_acc = np.zeros((rows,cols))
    knn_holdout_avg_acc = np.zeros((rows,cols))
    knn_test_avg_acc = np.zeros((rows,cols))

    #confusion matrices
    num_labels = len(labelset)
    num_labels_no_intro = len(non_intro_note_labelset)
    linsvm_holdout_acc_by_label = np.zeros((rows,cols,num_labels))
    linsvm_holdout_cm_arr = np.empty((rows,cols),dtype='O')
    linsvm_test_acc_by_label = np.zeros((rows,cols,num_labels))
    linsvm_test_cm_arr = np.empty((rows,cols),dtype='O')
    linsvm_holdout_no_intro_acc_by_label = np.zeros((rows,cols,num_labels_no_intro))
    linsvm_holdout_no_intro_cm_arr = np.empty((rows,cols),dtype='O')
    linsvm_test_no_intro_acc_by_label = np.zeros((rows,cols,num_labels_no_intro))
    linsvm_test_no_intro_cm_arr = np.empty((rows,cols),dtype='O')
    svm_holdout_acc_by_label = np.zeros((rows,cols,num_labels))
    svm_holdout_cm_arr = np.empty((rows,cols),dtype='O')
    svm_test_acc_by_label = np.zeros((rows,cols,num_labels))
    svm_test_cm_arr = np.empty((rows,cols),dtype='O')
    knn_holdout_acc_by_label = np.zeros((rows,cols,num_labels))
    knn_holdout_cm_arr = np.empty((rows,cols),dtype='O')
    knn_test_acc_by_label = np.zeros((rows,cols,num_labels))
    knn_test_cm_arr = np.empty((rows,cols),dtype='O')

    # get labels from data to calculate accuracy and confusion matrices
    svm_train_labels = load_from_mat(svm_train_fname)[1] # [1] because don't need samples or song_IDs returned by load_from_mat
    svm_train_labels = filter_labels(svm_train_labels,labelset)
    svm_test_labels = load_from_mat(svm_test_fname)[1]
    svm_test_labels = filter_labels(svm_test_labels,labelset)
    svm_test_labels_no_intro = filter_labels(svm_test_labels,non_intro_note_labelset)  
    knn_train_labels = load_knn_data(knn_train_fname,labelset)[1]
    knn_test_labels = load_knn_data(knn_test_fname,labelset)[1] 
    assert np.array_equal(knn_train_labels,svm_train_labels)

    ### loop that opens shelve database file for each replicate and puts values into summary data matrices
    for col_ind, num_songs in enumerate(NUM_SONGS_TO_TEST):
        for row_ind,replicate in enumerate(REPLICATES):
            shelve_fname = RESULTS_DIR + SHELVE_BASE_FNAME + str(bird_ID) + ", " + str(num_songs) + ' songs, replicate ' + str(replicate) + '.db'
            with shelve.open(shelve_fname,'r') as shv:
                # get number of samples, duration of samples 
                train_sample_IDs = shv['train_sample_IDs']
                num_train_samples[row_ind,col_ind] = len(train_sample_IDs)
                train_sample_total_duration[row_ind,col_ind] = shv['train_sample_total_duration']
                train_sample_no_intro_total_duration[row_ind,col_ind] = shv['train_sample_no_intro_total_duration']
                # need holdout sample indices to determine accuracy on holdout sets
                holdout_sample_IDs = shv['holdout_sample_IDs']
                holdout_labels = svm_train_labels[holdout_sample_IDs]
                holdout_labels_no_intro = filter_labels(holdout_labels,non_intro_note_labelset) #removes intro notes
                
                # put Rand accuracies in summary data matrices
                # below, [0] at end of line because liblinear Python API returns 3-element tuple, 1st element is acc.
                linsvm_holdout_rnd_acc[row_ind,col_ind] = shv['linsvm_holdout_acc'][0]  
                linsvm_test_rnd_acc[row_ind,col_ind] = shv['linsvm_test_acc'][0]
                linsvm_holdout_no_intro_rnd_acc[row_ind,col_ind] = \
                    shv['linsvm_holdout_no_intro_acc'][0]
                linsvm_test_no_intro_rnd_acc[row_ind,col_ind] = \
                    shv['linsvm_test_no_intro_acc'][0]
                svm_holdout_rnd_acc[row_ind,col_ind] = shv['svm_holdout_score'] * 100
                svm_test_rnd_acc[row_ind,col_ind] = shv['svm_test_score'] * 100
                knn_holdout_rnd_acc[row_ind,col_ind] = shv['knn_holdout_score'] * 100
                knn_test_rnd_acc[row_ind,col_ind] = shv['knn_test_score'] * 100

                # put average per-label accuracies in summary data matrices
                # and make confusion matrices
                linsvm_holdout_pred_labels = shv['linsvm_holdout_pred_labels']
                #have to convert to array since liblinear Python interface returns a list
                linsvm_holdout_pred_labels = np.asarray(linsvm_holdout_pred_labels,dtype='uint32')
                acc_by_label,avg_acc = get_acc_by_label(holdout_labels,linsvm_holdout_pred_labels,labelset)
                linsvm_holdout_acc_by_label[row_ind,col_ind] = acc_by_label * 100
                linsvm_holdout_avg_acc[row_ind,col_ind] = avg_acc * 100
                linsvm_holdout_confuse_mat = confuse_mat(holdout_labels,linsvm_holdout_pred_labels,labels=labelset)
                linsvm_holdout_cm_arr[row_ind,col_ind]  = linsvm_holdout_confuse_mat

                linsvm_test_pred_labels = shv['linsvm_test_pred_labels']
                pdb.set_trace()
                #have to convert to array since liblinear Python interface returns a list
                linsvm_test_pred_labels = np.asarray(linsvm_test_pred_labels,dtype='uint32')
                try:
                    acc_by_label,avg_acc = get_acc_by_label(svm_test_labels,linsvm_test_pred_labels,labelset)
                except IndexError:
                    pdb.set_trace()
                linsvm_test_acc_by_label[row_ind,col_ind] = acc_by_label * 100
                linsvm_test_avg_acc[row_ind,col_ind] = avg_acc * 100
                linsvm_test_confuse_mat = confuse_mat(svm_test_labels,linsvm_test_pred_labels,labels=labelset)
                linsvm_test_cm_arr[row_ind,col_ind] = linsvm_test_confuse_mat

                linsvm_holdout_no_intro_pred_labels = shv['linsvm_holdout_no_intro_pred_labels']
                #have to convert to array since liblinear Python interface returns a list
                linsvm_holdout_no_intro_pred_labels = np.asarray(linsvm_holdout_no_intro_pred_labels,dtype='uint32')
                acc_by_label,avg_acc = get_acc_by_label(holdout_labels_no_intro,linsvm_holdout_no_intro_pred_labels,non_intro_note_labelset)
                linsvm_holdout_no_intro_acc_by_label[row_ind,col_ind] = acc_by_label * 100
                linsvm_holdout_no_intro_avg_acc[row_ind,col_ind] = avg_acc * 100
                linsvm_holdout_no_intro_confuse_mat = confuse_mat(holdout_labels_no_intro,
                                                                linsvm_holdout_no_intro_pred_labels,
                                                                labels=non_intro_note_labelset)
                linsvm_holdout_no_intro_cm_arr[row_ind,col_ind]  = linsvm_holdout_no_intro_confuse_mat
                
                linsvm_test_no_intro_pred_labels = shv['linsvm_test_no_intro_pred_labels']
                #have to convert to array since liblinear Python interface returns a list
                linsvm_test_no_intro_pred_labels = np.asarray(linsvm_test_no_intro_pred_labels,dtype='uint32')
                acc_by_label,avg_acc = get_acc_by_label(svm_test_labels_no_intro,linsvm_test_no_intro_pred_labels,non_intro_note_labelset)
                linsvm_test_no_intro_acc_by_label[row_ind,col_ind] = acc_by_label * 100
                linsvm_test_no_intro_avg_acc[row_ind,col_ind] = avg_acc * 100
                linsvm_test_no_intro_confuse_mat = confuse_mat(svm_test_labels_no_intro,
                                                            linsvm_test_no_intro_pred_labels,
                                                            labels=non_intro_note_labelset)
                linsvm_test_no_intro_cm_arr[row_ind,col_ind] = linsvm_test_no_intro_confuse_mat

                svm_holdout_pred_labels = shv['svm_holdout_pred_labels']
                acc_by_label,avg_acc = get_acc_by_label(holdout_labels,svm_holdout_pred_labels,labelset)
                svm_holdout_acc_by_label[row_ind,col_ind] = acc_by_label * 100
                svm_holdout_avg_acc[row_ind,col_ind] = avg_acc * 100
                svm_holdout_confuse_mat = confuse_mat(holdout_labels,svm_holdout_pred_labels,labels=labelset)
                svm_holdout_cm_arr[row_ind,col_ind]  = svm_holdout_confuse_mat
                
                svm_test_pred_labels = shv['svm_test_pred_labels']
                acc_by_label,avg_acc = get_acc_by_label(svm_test_labels,svm_test_pred_labels,labelset)
                svm_test_acc_by_label[row_ind,col_ind] = acc_by_label * 100
                svm_test_avg_acc[row_ind,col_ind] = avg_acc * 100
                svm_test_confuse_mat = confuse_mat(svm_test_labels,svm_test_pred_labels,labels=labelset)
                svm_test_cm_arr[row_ind,col_ind] = svm_test_confuse_mat
                                
                knn_holdout_pred_labels = shv['knn_holdout_pred_labels']
                acc_by_label,avg_acc = get_acc_by_label(holdout_labels,knn_holdout_pred_labels,labelset)
                knn_holdout_acc_by_label[row_ind,col_ind] = acc_by_label * 100
                knn_holdout_avg_acc[row_ind,col_ind] = avg_acc * 100
                knn_holdout_confuse_mat = confuse_mat(holdout_labels,knn_holdout_pred_labels,labels=labelset)
                knn_holdout_cm_arr[row_ind,col_ind] = knn_holdout_confuse_mat
                                
                knn_test_pred_labels = shv['knn_test_pred_labels']
                acc_by_label,avg_acc = get_acc_by_label(knn_test_labels,knn_test_pred_labels,labelset)
                knn_test_acc_by_label[row_ind,col_ind] = acc_by_label * 100
                knn_test_avg_acc[row_ind,col_ind] = avg_acc * 100
                knn_test_confuse_mat = confuse_mat(knn_test_labels,knn_test_pred_labels,labels=labelset)
                knn_test_cm_arr[row_ind,col_ind] = knn_test_confuse_mat
            shv.close()

    results_fname = RESULTS_DIR + bird_ID + RESULTS_SHV_BASE_FNAME
    # now put all the summary data matrices in a summary data shelve database
    with shelve.open(results_fname) as shv:
        shv['train_sample_total_duration'] = train_sample_total_duration
        shv['num_train_samples'] = num_train_samples
        shv['labelset'] = labelset
        shv['non_intro_note_labelset'] = non_intro_note_labelset

        shv['linsvm_holdout_rnd_acc'] = linsvm_holdout_rnd_acc
        shv['linsvm_holdout_rnd_acc_mn'] = np.mean(linsvm_holdout_rnd_acc,axis=0)
        shv['linsvm_holdout_rnd_acc_std'] = np.std(linsvm_holdout_rnd_acc,axis=0)

        shv['linsvm_test_rnd_acc'] = linsvm_test_rnd_acc
        shv['linsvm_test_rnd_acc_mn'] = np.mean(linsvm_test_rnd_acc,axis=0)
        shv['linsvm_test_rnd_acc_std'] = np.std(linsvm_test_rnd_acc,axis=0)

        shv['linsvm_holdout_no_intro_rnd_acc'] = linsvm_holdout_no_intro_rnd_acc
        shv['linsvm_holdout_no_intro_rnd_acc_mn'] = np.mean(linsvm_holdout_no_intro_rnd_acc,axis=0)
        shv['linsvm_holdout_no_intro_rnd_acc_std'] = np.std(linsvm_holdout_no_intro_rnd_acc,axis=0)

        shv['linsvm_test_no_intro_rnd_acc'] = linsvm_test_no_intro_rnd_acc
        shv['linsvm_test_no_intro_rnd_acc_mn'] = np.mean(linsvm_test_no_intro_rnd_acc,axis=0)
        shv['linsvm_test_no_intro_rnd_acc_std'] = np.std(linsvm_test_no_intro_rnd_acc,axis=0)

        #for linear svm only, bin accuracy by number of samples, then get mean and std. dev.
        #only take accuracy from test set, to compare directly with Tachibana
        linsvm_test_rnd_acc_flat = linsvm_test_rnd_acc.flatten()
        num_train_samples_flat = num_train_samples.flatten()
        max_n_samples = np.max(num_train_samples_flat)
        max_n_rounded_to_nearest_1k = np.ceil(max_n_samples/1000).astype(int) * 1000
        BINS = list(range(0,max_n_rounded_to_nearest_1k,500))
        indices = np.digitize(num_train_samples_flat,BINS)
        bin_means = [linsvm_test_rnd_acc_flat[indices == i].mean() for i in range(1, len(BINS))]
        bin_std = [linsvm_test_rnd_acc_flat[indices == i].std() for i in range(1, len(BINS))]
        shv['linsvm_test_rnd_acc_flat'] = linsvm_test_rnd_acc_flat
        shv['num_train_samples_flat'] = num_train_samples_flat
        shv['linsvm_train_sample_bins'] = BINS
        shv['linsvm_test_rnd_acc_by_sample_mn'] = bin_means
        shv['linsvm_test_rnd_acc_by_sample_std'] = bin_std

        shv['svm_holdout_rnd_acc'] = svm_holdout_rnd_acc
        shv['svm_holdout_rnd_acc_mn'] = np.mean(svm_holdout_rnd_acc,axis=0)
        shv['svm_holdout_rnd_acc_std'] = np.std(svm_holdout_rnd_acc,axis=0)

        shv['svm_test_rnd_acc'] = svm_test_rnd_acc
        shv['svm_test_rnd_acc_mn'] = np.mean(svm_test_rnd_acc,axis=0)
        shv['svm_test_rnd_acc_std'] = np.std(svm_test_rnd_acc,axis=0)
        
        shv['knn_holdout_rnd_acc'] = knn_holdout_rnd_acc
        shv['knn_holdout_rnd_acc_mn'] = np.mean(knn_holdout_rnd_acc,axis=0)
        shv['knn_holdout_rnd_acc_std'] = np.std(knn_holdout_rnd_acc,axis=0)

        shv['knn_test_rnd_acc'] = knn_test_rnd_acc
        shv['knn_test_rnd_acc_mn'] = np.mean(knn_test_rnd_acc,axis=0)
        shv['knn_test_rnd_acc_std'] = np.std(knn_test_rnd_acc,axis=0)

        shv['linsvm_holdout_acc_by_label'] = linsvm_holdout_acc_by_label
        shv['linsvm_holdout_avg_acc'] = linsvm_holdout_avg_acc
        shv['linsvm_holdout_avg_acc_mn'] = np.mean(linsvm_holdout_avg_acc,axis=0)
        shv['linsvm_holdout_avg_acc_std'] = np.std(linsvm_holdout_avg_acc,axis=0)
        shv['linsvm_holdout_cm_arr'] = linsvm_holdout_cm_arr

        shv['linsvm_test_acc_by_label'] = linsvm_test_acc_by_label
        shv['linsvm_test_avg_acc'] = linsvm_test_avg_acc
        shv['linsvm_test_avg_acc_mn'] = np.mean(linsvm_test_avg_acc,axis=0)
        shv['linsvm_test_avg_acc_std'] = np.std(linsvm_test_avg_acc,axis=0)
        shv['linsvm_test_cm_arr'] = linsvm_test_cm_arr

        shv['linsvm_holdout_no_intro_acc_by_label'] = linsvm_holdout_no_intro_acc_by_label
        shv['linsvm_holdout_no_intro_avg_acc'] = linsvm_holdout_no_intro_avg_acc
        shv['linsvm_holdout_no_intro_avg_acc_mn'] = np.mean(linsvm_holdout_no_intro_avg_acc,axis=0)
        shv['linsvm_holdout_no_intro_avg_acc_std'] = np.std(linsvm_holdout_no_intro_avg_acc,axis=0)
        shv['linsvm_holdout_no_intro_cm_arr'] = linsvm_holdout_no_intro_cm_arr

        shv['linsvm_test_no_intro_acc_by_label'] = linsvm_test_no_intro_acc_by_label
        shv['linsvm_test_no_intro_avg_acc'] = linsvm_test_no_intro_avg_acc
        shv['linsvm_test_no_intro_avg_acc_mn'] = np.mean(linsvm_test_no_intro_avg_acc,axis=0)
        shv['linsvm_test_no_intro_avg_acc_std'] = np.std(linsvm_test_no_intro_avg_acc,axis=0)
        shv['linsvm_test_no_intro_cm_arr'] = linsvm_test_no_intro_cm_arr

        shv['svm_holdout_acc_by_label'] = svm_holdout_acc_by_label
        shv['svm_holdout_avg_acc'] = svm_holdout_avg_acc
        shv['svm_holdout_avg_acc_mn'] = np.mean(svm_holdout_avg_acc,axis=0)
        shv['svm_holdout_avg_acc_std'] = np.std(svm_holdout_avg_acc,axis=0)
        shv['svm_holdout_cm_arr'] = svm_holdout_cm_arr

        shv['svm_test_acc_by_label'] = svm_test_acc_by_label
        shv['svm_test_avg_acc'] = svm_test_avg_acc
        shv['svm_test_avg_acc_mn'] = np.mean(svm_test_avg_acc,axis=0)
        shv['svm_test_avg_acc_std'] = np.std(svm_test_avg_acc,axis=0)
        shv['svm_test_cm_arr'] = svm_test_cm_arr

        shv['knn_holdout_acc_by_label'] = knn_holdout_acc_by_label
        shv['knn_holdout_avg_acc'] = knn_holdout_avg_acc
        shv['knn_holdout_avg_acc_mn'] = np.mean(knn_holdout_avg_acc,axis=0)
        shv['knn_holdout_avg_acc_std'] = np.std(knn_holdout_avg_acc,axis=0)
        shv['knn_holdout_cm_arr'] = knn_holdout_cm_arr
      
        shv['knn_test_acc_by_label'] = knn_test_acc_by_label
        shv['knn_test_avg_acc'] = knn_test_avg_acc
        shv['knn_test_avg_acc_mn'] = np.mean(knn_test_avg_acc,axis=0)
        shv['knn_test_avg_acc_std'] = np.std(knn_test_avg_acc,axis=0)
        shv['knn_test_cm_arr'] = knn_test_cm_arr
    shv.close()
