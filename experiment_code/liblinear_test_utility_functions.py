import pdb
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

def filter_samples(samples,labels,labels_to_keep):
    """
    filter_samples(samples,labels,labels_to_keep)
        input parameters:
            samples -- list of dictionaries returned by svm_read_problem
                function from liblinear python interface
                (each dictionary in the list is a feature vector extracted from
                 a given sample)
            labels -- list of labels that corresponds to the list of samples
                also returned by svm_read_problem function
            labels_to_keep -- filter_samples finds all indices of a label that
                occurs in 'labels *and* in 'labels_to_keep', and keeps only those,
                filtering out every other label
        returns: filtered_samples,filtered_labels
    """
    indices = [index for index, val in enumerate(labels) if val in labels_to_keep]
    filtered_labels = [labels[index] for index in indices]
    filtered_samples = [samples[index] for index in indices]
    return filtered_labels, filtered_samples

def remove_samples_by_label(samples,labels,labels_to_remove=['0','-','i']):
    """
    remove_samples_by_label(samples,labels,labels_to_remove)
        input arguments:
            samples -- list of dictionaries returned by svm_read_problem
                function from liblinear python interface
                (each dictionary in the list is a feature vector extracted from
                 a given sample)
            labels -- list of integers representing labels
                (converted from char to int)
            labels_to_remove -- list of labels to find in 
        returns:
            filtered_labels -- list without elements in labels_to_remove
            filtered_samples -- list of dictionaries with dictionaries removed
                corresponding to elements removed from labels
    """
    indices = [index for index, val in enumerate(labels) if val not in labels_to_remove]
    filtered_labels = [labels[index] for index in indices]
    filtered_samples = [samples[index] for index in indices]
    return filtered_labels, filtered_samples

def array_to_dictlist(array):
    dictlist = [{j+1:array[i,j] for j in range(array.shape[1])} for i in range(array.shape[0])]
    return dictlist

def dictlist_to_array(dictlist):
    data = [[i[j] for j in i] for i in dictlist]
    array = np.asarray(data)
    return array
    
def scale_data(data,scal_factors=None):
    """
    scale_data(data,scal_factors)
    Scales data, where data is a list of dictionaries returned by svm_read_problem,
    by subtracting off mean of each integer key across dicts, then dividing by
    standard deviation, i.e., z-standardization.
        scal_factors: optional dictionary containing the keys 'column_mean' and
        'column_std', the mean and standard deviation values that should be used
        to scale. Can be used to apply the same factors to training and testing
        data. The dictionary is returned by the function so it can be obtained
        when you run scale_data on training and then passed along with the 
        testing data / data to be predicted.
    returns:
        z-standardized data: list of dictionaries with values now scaled
        scal_factors: automatically returned if these were not passed to the function
    """
    
    data_list = [[i[j] for j in i] for i in data]
    data_array = np.asarray(data_list)
    if scal_factors == None:
        return_scal_factors = True
        scal_factors = {}
        scal_factors['column_mean'] = np.mean(data_array,0)
        scal_factors['column_std'] = np.std(data_array,0)
    else:
        return_scal_factors = False
    z_standardized_data = \
        (data_array - scal_factors['column_mean']) / scal_factors['column_std']
    z_standardized_data = array_to_dictlist(z_standardized_data)
    if return_scal_factors:
        return z_standardized_data, scal_factors
    else:
        return z_standardized_data

def save_data(y,x,save_fname):
    num_samples = len(y)
    num_features =  len(x[1])
    with open(save_fname,"w") as f:
        for sample in range(0,num_samples):
            f.write(str(y[sample]) + " ")
            for feature in range(1,num_features+1):
                value = x[sample][feature]
                f.write(str(feature) + ":" + str(value) + " ")
            f.write("\n")

def find_label_ids(label_list,labels_to_find):
    """
    find_label_ids(label_list,labels_to_find)
        returns:
            label_ids -- a numpy array of indices corresponding to the locations
            in label_list of the labels in labels_to_find

        input arguments:
            label_list -- list of integer values
                that are ascii codes corresponding to song syllable labels
            labels_to_find -- list of chars that are the labels to find
                (this function converts the chars to ascii codes itself)
    """
    label_list = np.asarray(label_list)
    labels_to_find = [ord(label) for label in labels_to_find] # convert from char to ascii code  
    label_ids = [np.where(label_list == label) for label in labels_to_find]
    label_ids = np.asarray(label_ids)
    label_ids = np.concatenate(label_ids[:])
    label_ids = np.sort(label_ids)
    return label_ids

def get_model_from_file(model_filename):
    """
    get_model_from_file(model_filename)
        returns:
            model, a dictionary in which each key is a label, identified by the
            liblinear package, and the value for that key is a weight vector, w.
    (For multi-class svm, each class represented by a label has a corresponding
    vector obtained by training the model with that label as the positive sample
    and all other labels as the negative sample)
    """
    with open(model_filename, 'rU') as filename:
        lines = filename.readlines()

    nr_class = lines[1]
    nr_class = str.split(nr_class)
    nr_class = nr_class[-1]
    nr_class = int(nr_class)

    labels = lines[2]
    labels = str.split(labels)
    labels = labels[1:] # discard 0, element of list that just says "label"
    labels = [int(label) for label in labels]

    nr_feature = lines[3]
    nr_feature = str.split(nr_feature)
    nr_feature = nr_feature[-1]
    nr_feature = int(nr_feature)

    # "header" is lines 0-5. Lines 6:end contain values for w vector
    w = lines[6:] # i.e., keep from line 6 to the end
    w = [line.strip('\n') for line in w] # get rid of newline
    w = [str.split(line) for line in w] #split by whitespace
    w = list(chain.from_iterable(w)) #flatten list
    w = [float(element) for element in w]

    err_str = "Cannot evenly divide %r element vector into %r %r-element feature vectors"
    assert not len(w) % nr_feature == nr_class, \
         err_str % (len(w),nr_feature,nr_class)
    ids = range(0,len(w)+nr_feature,nr_feature)
    model = {}
    for ind,label in enumerate(labels):
        model[label] = w[ids[ind]:ids[ind+1]]
    return model
    
def fetch_feature_subset(feature_set,feature_vec):
    """
    fetch_feature_subset(feature_set,feature_vec)
        feature_set: numpy array with 532 columns, each an element of a 'feature'
        For more detail on each feature and how much each elements consists of,
        see 'makeAllFeatures.m'
        feature_vec: list representing features requested, e.g., [1 2 24]
        returns --> an array that is a subset of the original
    Some features occupy more than one element, e.g.,feature 1 is a spectrum
    with 128 elements. So:
        x_subset = fetch_feature_subset(x,[1])
    where x is a n x 532 array will return an n x 128 array.
 """
    
    feature_vec.sort()
    #check to make sure that feature_vec is valid!
    if feature_vec[-1] > 24:
        print("Feature_vector values should not exceed 24")
        return None
    ids = [] # initialize list
    for feature in feature_vec:
        if feature == 1:
            ids += list(range(0,129))
        elif feature == 2:
            ids += list(range(129,257))
        elif feature == 3:
            ids += list(range(257,385))        
        elif feature == 4:
            ids += list(range(385,513))
        else:
            ids += 513 + feature - 5

    return feature_set[:,ids]
