#feature extraction code

This folder consists of Matlab and Python code used to extract features for use with machine learning algorithms as described in "Comparison of machine learning algorithms applied to birdsong element classification". The code is applied to audio files produced by custom Labview software. Example audio files are found here: ["example day of song"](https://drive.google.com/folderview?id=0B0BKW2mh0ySnYWhkYmV6WnNFQ1U&usp=sharing)

## svm
### features used for support vector machine experiments
This folder contains the code used to extract features used with the support vector machine algorithms. The Matlab script "makeAllFeatures" was written by R.O. Tachibana and calculates the features used in Tachibana et al. 2014. The "make_features_for_svm.m" script is simply a wrapper around makeAllFeatures that loops through all the files in a directory. It runs the Tachibana script on all syllables in a file. After looping through all the files, it generates an output file with an array containing all the features from all audio files in the directory.
The file "extract_Tachibana_features.py" is an attempt to translate the "makeAllFeatures" file to Python.
So to reproduce the file 'gy6or6_feature_file_from_03-24-12_generated_02-14-16_23-05.mat' you would just type the following
```matlab
>>make_feature_file_for_svm
```

## knn
### features used for k-Nearest Neighbor experiments
This folder contains the Matlab code used to extract featuers used with the k-Nearest Neighbor algorithm. The Matlab scripts in this folder should be run on a directory of song files in the following order:
1.make_spect_files
 * generates spectrograms of each segmented syllable with "spect_from_rawsyl" function. This produces .cbin.spect files  
 *Note that for the experiments in this paper I used 8 ms windows with 50% overlap.*  
2.make_feature_files
 * loops through .spect files and calculates features, saves them to .cbin.ftr
3.concat_ftrs
 * loops through .cbin.ftr files and concatenates each feature vector into a giant array

Hence, using the [example audo files](https://drive.google.com/folderview?id=0B0BKW2mh0ySnYWhkYmV6WnNFQ1U&usp=sharing), you should be able to reproduce the feature file 'gy6or6_ftr_cell_03242012_generated_12152015.mat' by entering the following lines into Matlab:
```matlab
>>make_spect_files(8,0.5)
>>make_feature_files
>>concat_ftrs
```
