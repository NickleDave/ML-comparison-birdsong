% Create feature files from each day of song, then concatenate into one
% file. Assumes you have directory structure set up so each folder is a day of
% song.

while 1
    prompt = 'Select folders that contain labeled song to generate feature files';
    dirs = uipickfiles('Prompt',prompt);
   
    is_a_dir = zeros(numel(dirs),1);
    for i=1:numel(dirs)
        is_a_dir(i) = isdir(dirs{i});
    end
    
    if sum(is_a_dir) == numel(dirs)
        ONLY_DIRS_SELECTED_DIRS = true;
    end
    
    if ONLY_DIRS_SELECTED_DIRS == true
        break
    end
    
end

pause(2) % pause gives uipickfiles window time to close
% without pause it stays open and maintains focus, so you can't Ctrl-C
% to quit the main loop of this script

%constants -- parameters for make_spect_files
DURATION = 8; % milliseconds
OVERLAP = 0.5; % i.e., 50%, half of window overlaps
QUANTIFY_DELTAS = true; % for make_ftr_files_for_knn

root_dir = pwd;
for i=1:numel(dirs)
    cd(dirs{i})
    disp(['Change to directory ' dirs(i)])
    
    disp('Making feature files for k-NN')
    % feature files for k-NN
    make_spect_files_for_knn(DURATION,OVERLAP)
    make_feature_files_for_knn(QUANTIFY_DELTAS)
    concat_knn_ftrs
    ftr_file = ls('*_knn_ftr_file_*');
    copyfile(ftr_file,root_dir)
    
    disp('Making feature files for SVM')
    % feature files for SVM
    make_feature_file_for_svm
    ftr_file = ls('*_svm_ftr_file_from_*');
    copyfile(ftr_file,root_dir)
end

% then concatenate everything
cd(root_dir)
nowstr = datestr(now,'mm-dd-yy_HH-MM');
% for knn
knn_ftr_files = ls('*knn_ftr*');
all_ftr_cells = cell(2,12);
all_labels = [];
all_song_IDs = [];
all_dstrs = []; %
for i = 1:size(knn_ftr_files,1)
    load(knn_ftr_files(i,:));
    all_ftr_cells(1,:) = feature_cell(1,:);
    for j = 1:size(feature_cell,2)
        all_ftr_cells{2,j} = [all_ftr_cells{2,j}; feature_cell{2,j}];
    end
    all_labels = [all_labels;labels'];
    all_song_IDs = [all_song_IDs;song_IDs];
    all_dstrs = [all_dstrs;dstr];
end

feature_cell = all_ftr_cells;
labels = all_labels;
song_IDs = all_song_IDs;
save(['concat_knn_ftr_cell_generated_' nowstr '.mat'],'feature_cell','labels','song_IDs','all_dstrs')

% for svm
svm_ftr_files = ls('*svm_ftr*');
all_ftrs_mat = [];
all_labels_vec = [];
all_song_IDs_vec = [];
all_dstrs = []; %
for i = 1:size(svm_ftr_files,1)
    load(svm_ftr_files(i,:));
    all_ftrs_mat = [all_ftrs_mat;features_mat];
    all_labels_vec = [all_labels_vec;label_vec'];
    all_song_IDs_vec = [all_song_IDs_vec;song_IDs_vec];
    all_dstrs = [all_dstrs;dstr];
end
features_mat = all_ftrs_mat;
label_vec = all_labels_vec;
song_IDs_vec = all_song_IDs_vec;
save(['concat_svm_ftr_cell_generated_' nowstr '.mat'],'features_mat','label_vec','song_IDs_vec','all_dstrs')