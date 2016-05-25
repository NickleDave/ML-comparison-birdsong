feature extraction code:
========================

This folder consists of Matlab and Python code used to extract features for use with machine learning algorithms as described in "Comparison of machine learning algorithms applied to birdsong element classification". The code is applied to audio files produced by custom Labview software. Example audio files are found here: <hyperlink>.

svm
---
This folder contains the code used to extract features used with the support vector machine algorithms. The Matlab script "makeAllFeatures" was written by R.O. Tachibana and calculates the features used in Tachibana et al. 2014. The "make_features_for_svm.m" script is simply a wrapper around makeAllFeatures that loops through all the files in a directory. It runs the Tachibana script on all syllables in a file. After looping through all the files, it generates an output file with an array containing all the features from all audio files  in the directory.
The file "extract_Tachibana_features.py" is an attempt to translate the "makeAllFeatures" file to Python.

knn
---
This folder contains the Matlab code used to extract featuers used with the k-Nearest Neighbor algorithm. The Matlab scripts in this folder should be run on a directory of song files in the following order:
1.make_spect_files -- generates spectrograms of each segmented syllable with "spect_from_rawsyl" function. This produces .spect files 
2.make_feature_files -- loops through .spect files and calculates features
3.concat_ftrs -- loops

